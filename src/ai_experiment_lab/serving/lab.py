"""The model advisor endpoints and page (ADR-011).

``POST /lab/analyze``  raw CSV body -> columns and a suggested target (nothing is trained)
``POST /lab/run``      {upload_id, target, id_columns} -> a job id
``GET  /lab/jobs/{id}`` progress, then the recommendation
``GET  /lab/jobs/{id}/model`` the trained model as a zip

Uploads are held in memory only, for a short time, and are never written to disk or logged.
One recommendation runs at a time; every limit below can be changed with an environment variable
because a free server is far slower than a laptop.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ai_experiment_lab.automl import DatasetError, advise, column_profile, export_result
from ai_experiment_lab.automl.schema import slugify

log = logging.getLogger("ai_experiment_lab.lab")


def _env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


class Limits:
    """Read at app creation so tests and deployments can set them."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("LAB_ENABLED", "1") not in {"0", "false", "no"}
        self.max_upload = int(_env("LAB_MAX_UPLOAD_MB", 15) * 1024 * 1024)
        self.max_columns = int(_env("LAB_MAX_COLUMNS", 100))
        self.max_rows = int(_env("LAB_MAX_ROWS", 20000))          # rows used for training
        self.budget = _env("LAB_BUDGET_SECONDS", 60)             # time spent tuning
        self.hard = _env("LAB_HARD_SECONDS", 420)                # the job is stopped after this
        self.trials = int(_env("LAB_MAX_TRIALS", 10))
        self.ttl = _env("LAB_TTL_SECONDS", 1800)
        self.keep_uploads = 4


class RunRequest(BaseModel):
    upload_id: str = Field(min_length=1, max_length=64)
    target: str = Field(min_length=1, max_length=300)
    id_columns: list[str] = Field(default_factory=list, max_length=200)


class _Job:
    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.status = "queued"
        self.progress: list[str] = []
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.bundle: bytes | None = None
        self.bundle_name = "model.zip"
        self.created = time.time()
        self.started: float | None = None

    def view(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "status": self.status, "progress": self.progress[-30:]}
        if self.started:
            out["elapsed"] = round(time.time() - self.started, 1)
        if self.result is not None:
            out["result"] = self.result
            out["has_model"] = self.bundle is not None
        if self.error:
            out["error"] = self.error
        return out


class LabState:
    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.uploads: dict[str, tuple[pd.DataFrame, float]] = {}
        self.jobs: dict[str, _Job] = {}
        self.lock = threading.Lock()
        self.busy = False

    def sweep(self) -> None:
        cutoff = time.time() - self.limits.ttl
        for k in [k for k, (_, t) in self.uploads.items() if t < cutoff]:
            self.uploads.pop(k, None)
        for k in [k for k, j in self.jobs.items() if j.created < cutoff and j.status != "running"]:
            self.jobs.pop(k, None)
        while len(self.uploads) > self.limits.keep_uploads:
            self.uploads.pop(min(self.uploads, key=lambda k: self.uploads[k][1]), None)


def parse_csv(raw: bytes, limits: Limits) -> pd.DataFrame:
    """Read an uploaded CSV, or raise ``DatasetError`` saying what to fix."""
    if not raw.strip():
        raise DatasetError("The file is empty.")
    last: Exception | None = None
    df = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            break
        except Exception as exc:  # noqa: BLE001 - any parse failure becomes one clear message
            last = exc
    if df is None:
        raise DatasetError("This does not look like a CSV file (comma, semicolon or tab "
                           f"separated text with a header row). {type(last).__name__}")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [not c.startswith("Unnamed:") or df[c].notna().any() for c in df.columns]]
    if df.shape[1] < 2:
        raise DatasetError("The file needs at least two columns: one to predict and one or more "
                           "to predict it from.")
    if df.shape[1] > limits.max_columns:
        raise DatasetError(f"The file has {df.shape[1]} columns; this server accepts up to "
                           f"{limits.max_columns}.")
    if len(df) < 50:
        raise DatasetError(f"The file has {len(df)} rows; at least 50 are needed.")
    names = set(df.columns)
    # pandas renames a repeated header "a" to "a.1"; a column called "a.1" beside "a" is that.
    if any(c.rpartition(".")[0] in names and c.rpartition(".")[2].isdigit() for c in names):
        raise DatasetError("Two columns share a name; rename one and upload again.")
    return df


def _run_job(state: LabState, job: _Job, df: pd.DataFrame, req: RunRequest) -> None:
    lim = state.limits
    job.status, job.started = "running", time.time()

    def progress(message: str) -> None:
        if time.time() - job.started > lim.hard:
            raise TimeoutError
        job.progress.append(message)

    try:
        advice = advise(df, req.target, id_columns=req.id_columns or None,
                        budget_seconds=lim.budget, max_rows=lim.max_rows,
                        max_trials=lim.trials, progress=progress)
        if advice.result is not None:
            with tempfile.TemporaryDirectory() as tmp:
                name = slugify(f"{req.target}-advisor")
                folder = export_result(advice.result, Path(tmp) / "model", name,
                                       title=f"{req.target} predictor")
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in sorted(Path(folder).iterdir()):
                        zf.write(f, f"model/{f.name}")
                    zf.writestr("model/advice.json",
                                json.dumps(advice.to_dict(), indent=2, default=str))
                job.bundle = buf.getvalue()
        job.result = advice.to_dict()
        job.status = "done"
    except TimeoutError:
        job.status, job.error = "error", ("This took longer than the server allows. Try a "
                                          "smaller file or fewer columns.")
    except (DatasetError, ValueError) as exc:
        job.status, job.error = "error", str(exc)
    except Exception:
        log.exception("advisor job failed")
        job.status, job.error = "error", "Something went wrong while analysing the file."
    finally:
        with state.lock:
            state.busy = False


async def _read_body(request: Request, cap: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > cap:
        raise HTTPException(413, f"The file is larger than {cap // (1024 * 1024)} MB.")
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise HTTPException(413, f"The file is larger than {cap // (1024 * 1024)} MB.")
        chunks.append(chunk)
    return b"".join(chunks)


def register_lab(app: FastAPI) -> LabState | None:
    lim = Limits()
    if not lim.enabled:
        return None
    state = LabState(lim)
    app.state.lab = state
    page = render_lab(lim)

    @app.get("/lab", include_in_schema=False)
    def lab_page() -> HTMLResponse:
        return HTMLResponse(page)

    @app.get("/lab/limits")
    def lab_limits() -> dict[str, Any]:
        return {"max_upload_mb": lim.max_upload / 1024 / 1024, "max_columns": lim.max_columns,
                "rows_used_for_training": lim.max_rows, "max_trials": lim.trials,
                "tuning_seconds": lim.budget, "hard_stop_seconds": lim.hard,
                "kept_minutes": lim.ttl / 60}

    @app.post("/lab/analyze")
    async def analyze(request: Request) -> JSONResponse:
        raw = await _read_body(request, lim.max_upload)
        try:
            df = parse_csv(raw, lim)
        except DatasetError as exc:
            raise HTTPException(422, str(exc)) from exc
        with state.lock:
            state.sweep()
            upload_id = uuid.uuid4().hex
            state.uploads[upload_id] = (df, time.time())
        prof = column_profile(df)
        return JSONResponse({"upload_id": upload_id, **prof})

    @app.post("/lab/run", status_code=202)
    def run(req: RunRequest) -> dict[str, Any]:
        with state.lock:
            state.sweep()
            entry = state.uploads.get(req.upload_id)
            if entry is None:
                raise HTTPException(404, "That upload has expired. Upload the file again.")
            if state.busy:
                raise HTTPException(429, "Another analysis is running. Try again in a minute.")
            df = entry[0]
            if req.target not in df.columns:
                raise HTTPException(422, f"Column '{req.target}' is not in the file.")
            state.busy = True
            job = _Job(uuid.uuid4().hex)
            state.jobs[job.id] = job
        threading.Thread(target=_run_job, args=(state, job, df, req), daemon=True).start()
        return {"job_id": job.id}

    @app.get("/lab/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown or expired job.")
        return job.view()

    @app.get("/lab/jobs/{job_id}/model")
    def job_model(job_id: str) -> Response:
        job = state.jobs.get(job_id)
        if job is None or job.bundle is None:
            raise HTTPException(404, "No model is available for this job.")
        return Response(job.bundle, media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="model.zip"'})

    return state


def render_lab(lim: Limits) -> str:
    cfg = json.dumps({"maxMb": lim.max_upload / 1024 / 1024, "rows": lim.max_rows,
                      "seconds": lim.budget}).replace("<", "\\u003c")
    return _PAGE.replace("__CONFIG__", cfg)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model advisor</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --ink:#1b2430; --muted:#5d6b7c; --line:#dfe4ea; --accent:#1f5fbf; --ok:#17794a; --warn:#9a6700; --warnbg:#fff7e0; --bar:#8fb3e8; }
@media (prefers-color-scheme: dark) { :root { --bg:#12161c; --card:#1b212a; --ink:#e7ecf2; --muted:#9aa8b8; --line:#2c3541; --accent:#6ea3ff; --ok:#4cc38a; --warn:#e3b341; --warnbg:#2a2412; --bar:#33598f; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
header, main { max-width:900px; margin:0 auto; padding:0 16px; } header { padding-top:28px; }
h1 { font-size:1.6rem; margin:0 0 4px; } h2 { font-size:1.15rem; margin:0 0 10px; }
.sub, .small { color:var(--muted); } .small { font-size:.88rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; margin:16px 0; }
label { display:block; font-weight:600; margin:10px 0 4px; }
select, input[type=file] { width:100%; max-width:460px; padding:9px; font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line); border-radius:8px; }
button { font:inherit; font-weight:600; padding:10px 18px; border-radius:8px; border:1px solid var(--accent); background:var(--accent); color:#fff; cursor:pointer; }
button.alt { background:transparent; color:var(--accent); } button:disabled { opacity:.5; cursor:default; }
.ids { display:flex; flex-wrap:wrap; gap:6px 16px; margin:4px 0; } .ids label { font-weight:400; margin:0; display:flex; gap:6px; }
.error { color:#b42318; white-space:pre-line; margin-top:10px; } .note { background:var(--warnbg); border:1px solid var(--warn); border-radius:8px; padding:10px 12px; margin:8px 0; }
.big { font-size:1.7rem; font-weight:700; } .pill { display:inline-block; border:1px solid var(--ok); color:var(--ok); border-radius:999px; padding:1px 10px; font-size:.85rem; font-weight:600; margin-left:8px; vertical-align:middle; }
table { border-collapse:collapse; width:100%; font-size:.92rem; } th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; } td.n { text-align:right; font-variant-numeric:tabular-nums; }
.scroll { overflow-x:auto; } .bar { height:10px; background:var(--bar); border-radius:5px; } .barcell { width:40%; }
pre { background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:12px; overflow-x:auto; font-size:.85rem; }
ul { padding-left:20px; margin:6px 0; } [hidden] { display:none !important; }
progress { width:100%; max-width:460px; } footer { max-width:900px; margin:0 auto; padding:0 16px 32px; color:var(--muted); font-size:.85rem; }
a { color:var(--accent); }
</style></head><body>
<header>
  <h1>Model advisor</h1>
  <p class="sub">Upload a table of data (CSV). Pick the column you want to predict. The server tries several algorithms, tunes them, and tells you which to use and with what settings.</p>
  <p class="small"><a href="app">&larr; Back to the churn predictor</a></p>
</header>
<main>
  <section class="card" id="s-upload">
    <h2>1. Your data</h2>
    <label for="file">CSV file</label>
    <input type="file" id="file" accept=".csv,.tsv,.txt,text/csv">
    <p class="small" id="limits"></p>
    <p class="small">Works for tables where each row stands alone: predicting a category (yes/no or several classes) or a number. It does not handle time series (forecasting over time), images, audio or video.</p>
    <div class="error" id="e-upload" role="alert"></div>
  </section>

  <section class="card" id="s-setup" hidden>
    <h2>2. What should it predict?</h2>
    <p class="small" id="summary"></p>
    <label for="target">Column to predict</label>
    <select id="target"></select>
    <label>Columns to ignore (IDs, names, anything that is not a real input)</label>
    <div class="ids" id="ids"></div>
    <div class="note" id="p-warn" hidden></div>
    <p><button id="go">Find the best algorithm</button></p>
    <div class="error" id="e-run" role="alert"></div>
  </section>

  <section class="card" id="s-progress" hidden>
    <h2>Working&hellip;</h2>
    <progress id="prog"></progress>
    <p class="small" id="msg">Starting</p>
    <p class="small">This usually takes one to a few minutes on a small server. You can leave this tab open.</p>
  </section>

  <section id="s-result" hidden></section>
</main>
<footer>Your file is kept in the server's memory for 30 minutes at most, is not saved to disk and is not used for anything else. Results are guidance from a small, time-limited search; check them on your own data before relying on them.</footer>
<script>
"use strict";
const CFG = __CONFIG__;
const $ = id => document.getElementById(id);
function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) { if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v); }
  for (const k of kids) n.append(k && k.nodeType ? k : document.createTextNode(String(k ?? "")));
  return n;
}
$("limits").textContent = `Up to ${CFG.maxMb} MB. To stay fast on a free server, at most ${CFG.rows.toLocaleString()} rows are used for the search and about ${CFG.seconds} seconds are spent tuning.`;
let upload = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = null; try { body = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) { const d = body && body.detail; throw new Error(typeof d === "string" ? d : (d ? "The request was not valid." : "Request failed (" + r.status + ").")); }
  return body;
}

$("file").addEventListener("change", async ev => {
  const f = ev.target.files[0];
  $("e-upload").textContent = ""; $("s-setup").hidden = true; $("s-result").hidden = true; upload = null;
  if (!f) return;
  if (f.size > CFG.maxMb * 1024 * 1024) { $("e-upload").textContent = `That file is ${(f.size/1048576).toFixed(1)} MB; the limit is ${CFG.maxMb} MB.`; return; }
  try {
    const p = await api("lab/analyze", {method: "POST", headers: {"Content-Type": "text/csv"}, body: f});
    upload = p; fillSetup(p, f.name);
  } catch (e) { $("e-upload").textContent = e.message; }
});

function fillSetup(p, name) {
  $("summary").textContent = `${name}: ${p.rows.toLocaleString()} rows, ${p.columns.length} columns.`;
  const t = $("target"); t.replaceChildren();
  for (const c of p.columns) t.append(el("option", {value: c.name, text: `${c.name}  (${c.unique} distinct${c.missing ? ", " + c.missing + " missing" : ""})`}));
  t.value = p.suggested_target;
  renderIds();
  const w = $("p-warn"); w.hidden = !p.warnings.length; w.textContent = p.warnings.join(" ");
  $("s-setup").hidden = false; $("e-run").textContent = "";
}
function renderIds() {
  const box = $("ids"); box.replaceChildren();
  const ticked = new Set(upload.suggested_ids);
  for (const c of upload.columns) {
    if (c.name === $("target").value) continue;
    const cb = el("input", {type: "checkbox", value: c.name}); cb.checked = ticked.has(c.name);
    box.append(el("label", {}, cb, c.name));
  }
}
$("target").addEventListener("change", renderIds);

$("go").addEventListener("click", async () => {
  $("e-run").textContent = ""; $("go").disabled = true;
  const ids = [...$("ids").querySelectorAll("input:checked")].map(i => i.value);
  try {
    const r = await api("lab/run", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({upload_id: upload.upload_id, target: $("target").value, id_columns: ids})});
    $("s-progress").hidden = false; $("s-result").hidden = true; $("msg").textContent = "Starting";
    poll(r.job_id);
  } catch (e) { $("e-run").textContent = e.message; $("go").disabled = false; }
});

async function poll(id) {
  try {
    const j = await api("lab/jobs/" + id);
    if (j.progress && j.progress.length) $("msg").textContent = j.progress[j.progress.length - 1];
    if (j.status === "done") { $("s-progress").hidden = true; $("go").disabled = false; showResult(j, id); return; }
    if (j.status === "error") { $("s-progress").hidden = true; $("go").disabled = false; $("e-run").textContent = j.error; return; }
  } catch (e) { $("s-progress").hidden = true; $("go").disabled = false; $("e-run").textContent = e.message; return; }
  setTimeout(() => poll(id), 1500);
}

const METRICS = {
  binary: [["f1", "F1 (balance of precision and recall)"], ["roc_auc", "ROC AUC (ranking quality)"], ["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"]],
  multiclass: [["f1_macro", "Macro F1"], ["accuracy", "Accuracy"], ["balanced_accuracy", "Balanced accuracy"]],
  regression: [["rmse", "RMSE (typical error, lower is better)"], ["mae", "MAE (average error, lower is better)"], ["r2", "R² (share explained, higher is better)"]],
};
const fmt = v => (v === undefined || v === null || Number.isNaN(v)) ? "–" : (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3));

function table(head, rows, numCols) {
  const t = el("table"); const h = el("tr"); head.forEach(x => h.append(el("th", {text: x}))); t.append(h);
  for (const r of rows) { const tr = el("tr"); r.forEach((x, i) => tr.append(x && x.nodeType ? x : el("td", {class: numCols && numCols.includes(i) ? "n" : "", text: x}))); t.append(tr); }
  const s = el("div", {class: "scroll"}); s.append(t); return s;
}
function card(title, ...kids) { const c = el("section", {class: "card"}); if (title) c.append(el("h2", {text: title})); kids.forEach(k => c.append(k)); return c; }

function showResult(job, id) {
  const a = job.result, root = $("s-result"); root.replaceChildren();
  const big = el("div", {class: "big"}, prettyName(a.recommended), el("span", {class: "pill", text: a.task === "binary" ? "yes / no" : a.task === "multiclass" ? "several categories" : "a number"}));
  const head = card(null, el("div", {class: "small", text: "Recommended algorithm"}), big);
  const ul = el("ul"); a.why.forEach(w => ul.append(el("li", {text: w}))); head.append(ul);
  root.append(head);

  if (a.warnings.length) { const c = card("Double-check these"); a.warnings.forEach(w => c.append(el("div", {class: "note", text: w}))); root.append(c); }

  const pk = Object.keys(a.params);
  const pc = card("Settings to use");
  if (pk.length) pc.append(table(["Setting", "Value", "What it does"], pk.map(k => [k, String(a.params[k]), a.param_help[k] || ""])));
  else pc.append(el("p", {class: "small", text: "The algorithm's defaults worked best; no changes needed."}));
  root.append(pc);

  const ms = METRICS[a.task]; const sel = a.selection_metric;
  const sc = card("How well it does on rows it never saw");
  sc.append(table(["Measure", "This model", "Just guessing"], ms.map(([k, label]) => [label + (k === sel ? "  ← used to choose" : ""), fmt(a.metrics[k]), fmt(a.baseline[k])]), [1, 2]));
  root.append(sc);

  const lb = card("Every candidate tried");
  lb.append(table(["Algorithm", "Cross-validated score", "Score on unseen rows", "Fit time (s)"], a.leaderboard.map(r => [r.model, `${fmt(r.cv)} ± ${fmt(r.cv_std)}`, fmt(r.test), String(r.fit_seconds)]), [1, 2, 3]));
  lb.append(el("p", {class: "small", text: "Cross-validation uses only the training rows. A \"(tuned)\" row is the algorithm after a search for better settings; a tuned score is the best of several tries and can be a little optimistic, which is why the unseen-rows column is shown."}));
  root.append(lb);

  if (a.importances.length) {
    const mx = Math.max(...a.importances.map(i => i.share));
    const ic = card("What matters most in your data");
    ic.append(table(["Column", "Share of the model's skill", ""], a.importances.map(i => [i.feature, (i.share * 100).toFixed(0) + "%", el("td", {class: "barcell"}, (() => { const b = el("div", {class: "bar"}); b.style.width = (100 * i.share / mx) + "%"; return b; })())]), [1]));
    root.append(ic);
  }

  const cc = card("Use it yourself");
  cc.append(el("pre", {text: a.code}));
  const row = el("p");
  const cp = el("button", {class: "alt", text: "Copy code"}); cp.addEventListener("click", () => navigator.clipboard && navigator.clipboard.writeText(a.code).then(() => { cp.textContent = "Copied"; }));
  row.append(cp);
  if (job.has_model) { row.append(" "); const dl = el("a", {href: "lab/jobs/" + id + "/model", download: "model.zip"}); dl.append(el("button", {text: "Download the trained model"})); row.append(dl); }
  cc.append(row);
  cc.append(el("p", {class: "small", text: `Data used: ${a.data.used_rows.toLocaleString()} of ${a.data.rows.toLocaleString()} rows, ${a.data.features} input columns. Took ${a.seconds} s.`}));
  if (a.notes.length) { const n = el("ul"); a.notes.forEach(x => n.append(el("li", {class: "small", text: x}))); cc.append(n); }
  root.append(cc);
  root.hidden = false; root.scrollIntoView({behavior: "smooth", block: "start"});
}
function prettyName(k) { return ({logistic_regression: "Logistic regression", ridge: "Ridge regression", random_forest: "Random forest", xgboost: "XGBoost"})[k] || k; }
</script></body></html>
"""
