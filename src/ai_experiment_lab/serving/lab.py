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

    @app.get("/lab/uploads/{upload_id}")
    def upload_profile(upload_id: str) -> dict[str, Any]:
        """The profile of a file already uploaded (used when another page hands a file over)."""
        with state.lock:
            state.sweep()
            entry = state.uploads.get(upload_id)
        if entry is None:
            raise HTTPException(404, "That upload has expired. Upload the file again.")
        return {"upload_id": upload_id, **column_profile(entry[0])}

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
    from ai_experiment_lab.serving import theme
    return (_PAGE.replace("__THEME_CSS__", theme.CSS).replace("__FONTS__", theme.FONTS)
            .replace("__TOPBAR__", theme.topbar("advisor")).replace("__CONFIG__", cfg))


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model advisor</title>
__FONTS__
<style>
__THEME_CSS__
.card { background:var(--glass); border:1px solid var(--line); border-radius:18px; padding:clamp(16px,3vw,28px); margin-bottom:20px; }
.card > h2 { font-size:1.3rem; margin-bottom:14px; display:flex; align-items:center; gap:12px; }
.card > h2 .n { display:inline-grid; place-items:center; width:30px; height:30px; border-radius:50%; background:var(--reagent); color:var(--reagent-ink); font:700 .95rem var(--text); flex:none; }
.drop { position:relative; display:flex; flex-direction:column; align-items:center; gap:6px; margin-top:28px; padding:clamp(32px,6vw,56px) 16px; text-align:center; border:2px dashed rgba(232,245,242,.35); border-radius:20px; background:rgba(255,255,255,.06); color:var(--deep-ink); transition:border-color .15s, background .15s; cursor:pointer; }
.drop:hover, .drop.over { border-color:#8fe3bd; background:rgba(143,227,189,.12); }
.drop strong { font:800 clamp(1.4rem,3.5vw,2rem) var(--display); letter-spacing:-.015em; }
.drop span { color:var(--deep-muted); }
.trio { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:20px 28px; }
.trio strong { display:block; font:700 1.05rem var(--display); margin-bottom:4px; } .trio span { color:var(--muted); font-size:.94rem; }
#s-intro > h2 { font-size:1.3rem; margin-bottom:16px; }
.drop input { position:absolute; inset:0; width:100%; height:100%; opacity:0; cursor:pointer; padding:0; }
.drop:focus-within { outline:3px solid var(--focus); outline-offset:2px; }
.ids { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 4px; }
.ids label { flex-direction:row; align-items:center; gap:8px; font-weight:400; padding:6px 14px; border:1.5px solid var(--line); border-radius:999px; background:var(--sunk); cursor:pointer; transition:background .12s,border-color .12s; }
.ids label:has(input:checked) { background:color-mix(in srgb, var(--reagent) 14%, var(--glass)); border-color:var(--reagent); }
.ids input { accent-color:var(--reagent); }
.field { margin:16px 0 6px; font-weight:600; font-size:.92rem; }
#target { max-width:460px; }
.progress { height:8px; border-radius:4px; background:var(--sunk); overflow:hidden; border:1px solid var(--line); }
.progress i { display:block; height:100%; width:35%; border-radius:4px; background:linear-gradient(90deg,#2fbf84,#f0a21c,#ee3d7c); animation:slide 1.4s ease-in-out infinite alternate; }
@keyframes slide { from { margin-left:0; } to { margin-left:65%; } }
.winner { background:var(--deep); color:var(--deep-ink); border:0; }
.winner .small { color:var(--deep-muted); }
.winner .big { font:800 clamp(2.2rem,6vw,3.6rem)/1.05 var(--display); letter-spacing:-.025em; margin:6px 0 4px; display:flex; flex-wrap:wrap; align-items:center; gap:12px 18px; }
.winner .pill { font:600 .85rem var(--text); letter-spacing:0; padding:4px 14px; border-radius:999px; background:rgba(255,255,255,.14); }
.winner ul { margin:14px 0 0; padding-left:20px; max-width:70ch; } .winner li { margin:5px 0; }
.race { display:grid; gap:10px; }
.race .r { display:grid; grid-template-columns:minmax(120px,210px) 1fr 120px; gap:14px; align-items:center; font-size:.93rem; }
.race .name { font-weight:600; } .race .name small { display:block; color:var(--muted); font-weight:400; }
.race .track { height:14px; border-radius:7px; background:var(--sunk); border:1px solid var(--line); overflow:hidden; }
.race .track i { display:block; height:100%; border-radius:7px; background:color-mix(in srgb, var(--reagent) 45%, var(--sunk)); }
.race .r.best .track i { background:var(--reagent); } .race .r.best .name::after { content:" · best"; font-weight:500; color:var(--reagent); }
.race .v { text-align:right; font-variant-numeric:tabular-nums; }
.bar { height:10px; background:var(--reagent); border-radius:5px; } .barcell { width:40%; }
pre { background:var(--deep); color:var(--deep-ink); border-radius:12px; padding:16px; overflow-x:auto; font:.85rem/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; }
ul { padding-left:20px; margin:6px 0; }
.big { font-size:1.7rem; font-weight:700; }
footer { max-width:1040px; margin:0 auto; padding:0 20px 40px; color:var(--muted); font-size:.85rem; }
@media (max-width:640px) { .race .r { grid-template-columns:1fr 90px; } .race .track { grid-column:1 / -1; order:3; } }
</style></head><body>
__TOPBAR__
<div class="hero"><div class="in">
  <h1>Check your CSV. Find the right model.</h1>
  <p>Drop in any table of data. The advisor tries several algorithms on it, tunes them, and tells you which one to use and with what settings.</p>
  <label class="drop" id="drop"><strong id="drop-title">Drop your CSV here</strong><span id="drop-sub">or click to choose a file</span>
    <input type="file" id="file" accept=".csv,.tsv,.txt,text/csv" aria-label="CSV file"></label>
  <p class="hint2" id="limits"></p>
  <p class="hint2" style="margin-top:6px">Works for tables where each row stands alone: predicting a category (yes/no or several classes) or a number. It does not handle time series (forecasting over time), images, audio or video.</p>
  <div class="error" id="e-upload" role="alert" style="color:#ffb3c9"></div>
</div></div>
<main class="wrap">
  <section class="card" id="s-intro">
    <h2>What you get back</h2>
    <div class="trio">
      <div><strong>The algorithm and its settings</strong><span>One recommendation, with the reason and every setting explained.</span></div>
      <div><strong>An honest score</strong><span>Measured on rows the model never saw, next to what plain guessing would score.</span></div>
      <div><strong>A model you can use</strong><span>Download it as a zip and serve it with this project, or copy the code.</span></div>
    </div>
  </section>

  <section class="card" id="s-setup" hidden>
    <h2><span class="n">1</span>What should it predict?</h2>
    <p class="small" id="summary" style="margin-top:0"></p>
    <label for="target" class="field">Column to predict</label>
    <select id="target"></select>
    <div class="field">Columns to ignore (IDs, names, anything that is not a real input)</div>
    <div class="ids" id="ids"></div>
    <div class="note" id="p-warn" hidden></div>
    <p style="margin-bottom:0"><button class="primary" id="go">Find the best algorithm</button></p>
    <div class="error" id="e-run" role="alert"></div>
  </section>

  <section class="card" id="s-progress" hidden>
    <h2><span class="n">2</span>Trying algorithms&hellip;</h2>
    <div class="progress" id="prog" role="progressbar" aria-label="Working"><i></i></div>
    <p id="msg" style="margin:14px 0 4px;font-weight:500">Starting</p>
    <p class="small" style="margin:0">This usually takes one to a few minutes on a small server. You can leave this tab open.</p>
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

for (const ev of ["dragenter", "dragover"]) $("drop").addEventListener(ev, () => $("drop").classList.add("over"));
for (const ev of ["dragleave", "drop"]) $("drop").addEventListener(ev, () => $("drop").classList.remove("over"));
$("file").addEventListener("change", async ev => {
  const f = ev.target.files[0];
  $("drop-title").textContent = f ? f.name : "Drop your CSV here";
  $("drop-sub").textContent = f ? (f.size < 1048576 ? Math.round(f.size / 1024) + " KB" : (f.size / 1048576).toFixed(1) + " MB") + ", click to choose another" : "or click to choose a file";
  $("s-intro").hidden = !!f;
  $("e-upload").textContent = ""; $("s-setup").hidden = true; $("s-result").hidden = true; upload = null;
  if (!f) return;
  if (f.size > CFG.maxMb * 1024 * 1024) { $("e-upload").textContent = `That file is ${(f.size/1048576).toFixed(1)} MB; the limit is ${CFG.maxMb} MB.`; return; }
  try {
    const p = await api("lab/analyze", {method: "POST", headers: {"Content-Type": "text/csv"}, body: f});
    upload = p; fillSetup(p, f.name);
  } catch (e) { $("e-upload").textContent = e.message; }
});

(async () => {
  // Handed over from the churn page: #upload=<id>&name=<file name>
  const h = new URLSearchParams(location.hash.slice(1)); const id = h.get("upload");
  if (!id) return;
  history.replaceState(null, "", location.pathname);
  try { upload = await api("lab/uploads/" + encodeURIComponent(id)); fillSetup(upload, h.get("name") || "Your file"); }
  catch (e) { $("e-upload").textContent = e.message; }
})();

function fillSetup(p, name) {
  $("s-intro").hidden = true; $("drop-title").textContent = name;
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
  head.classList.add("winner");
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
  const lower = a.selection_metric === "rmse";
  const vals = a.leaderboard.map(r => r.cv), top = Math.max(...vals), low = Math.min(...vals);
  const race = el("div", {class: "race"});
  a.leaderboard.forEach((r, i) => {
    const w = lower ? (low > 0 ? low / r.cv : 1) : (top > 0 ? Math.max(0.03, r.cv / top) : 0);
    const bar = el("i"); bar.style.width = Math.max(3, Math.min(100, w * 100)) + "%";
    const track = el("div", {class: "track"}, bar);
    race.append(el("div", {class: "r" + (i === 0 ? " best" : "")},
      el("div", {class: "name"}, prettyName(r.model.replace(" (tuned)", "")) + (r.model.includes("(tuned)") ? " (tuned)" : ""), el("small", {text: `unseen rows ${fmt(r.test)}, fit ${r.fit_seconds} s`})),
      track, el("div", {class: "v", text: `${fmt(r.cv)} ± ${fmt(r.cv_std)}`})));
  });
  lb.append(race);
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
  const cp = el("button", {class: "secondary", text: "Copy code"}); cp.addEventListener("click", () => navigator.clipboard && navigator.clipboard.writeText(a.code).then(() => { cp.textContent = "Copied"; }));
  row.append(cp);
  if (job.has_model) { row.append(" "); const dl = el("a", {href: "lab/jobs/" + id + "/model", download: "model.zip"}); dl.append(el("button", {class: "primary", text: "Download the trained model"})); row.append(dl); }
  cc.append(row);
  cc.append(el("p", {class: "small", text: `Data used: ${a.data.used_rows.toLocaleString()} of ${a.data.rows.toLocaleString()} rows, ${a.data.features} input columns. Took ${a.seconds} s.`}));
  if (a.notes.length) { const n = el("ul"); a.notes.forEach(x => n.append(el("li", {class: "small", text: x}))); cc.append(n); }
  root.append(cc);
  root.hidden = false; root.scrollIntoView({behavior: "smooth", block: "start"});
}
function prettyName(k) { return ({logistic_regression: "Logistic regression", ridge: "Ridge regression", random_forest: "Random forest", xgboost: "XGBoost"})[k] || k; }
</script></body></html>
"""
