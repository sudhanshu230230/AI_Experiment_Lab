"""The web page served at ``/app``: a form for one customer and a CSV upload for many.

The form is generated from the same pydantic model the API validates with, so the page cannot
drift out of step with the API (a new category added to the schema appears in the dropdown).
Plain HTML and a little JavaScript: no build step, no extra dependency.
"""

from __future__ import annotations

import html
import json
from typing import Any

from ai_experiment_lab.serving.schemas import CustomerFeatures

# (label, help text) shown to people instead of the raw column names.
LABELS: dict[str, tuple[str, str]] = {
    "gender": ("Gender", ""),
    "SeniorCitizen": ("Senior citizen", ""),
    "Partner": ("Has a partner", ""),
    "Dependents": ("Has dependents", ""),
    "tenure": ("Months as a customer", "0 to 120"),
    "PhoneService": ("Phone service", ""),
    "MultipleLines": ("Multiple phone lines", ""),
    "InternetService": ("Internet service", ""),
    "OnlineSecurity": ("Online security add-on", ""),
    "OnlineBackup": ("Online backup add-on", ""),
    "DeviceProtection": ("Device protection add-on", ""),
    "TechSupport": ("Tech support add-on", ""),
    "StreamingTV": ("Streaming TV", ""),
    "StreamingMovies": ("Streaming movies", ""),
    "Contract": ("Contract", ""),
    "PaperlessBilling": ("Paperless billing", ""),
    "PaymentMethod": ("Payment method", ""),
    "MonthlyCharges": ("Monthly charge ($)", "0 to 1000"),
    "TotalCharges": ("Total charged so far ($)", "Leave blank for a brand-new customer"),
}
GROUPS: list[tuple[str, list[str]]] = [
    ("About the customer", ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"]),
    ("Services", ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                  "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                  "StreamingMovies"]),
    ("Contract and billing", ["Contract", "PaperlessBilling", "PaymentMethod",
                              "MonthlyCharges", "TotalCharges"]),
]
YES_NO_NUM = {0: "No", 1: "Yes"}

PRESETS: dict[str, dict[str, Any]] = {
    "new_flexible": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": "420.35",
    },
    "long_committed": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 60,
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
        "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 64.9,
        "TotalCharges": "3890.5",
    },
}


def _field_schema() -> dict[str, dict[str, Any]]:
    props = CustomerFeatures.model_json_schema()["properties"]
    return {name: props[name] for name in CustomerFeatures.model_fields}


def _control(name: str, spec: dict[str, Any]) -> str:
    label, hint = LABELS[name]
    hint_html = f'<span class="hint">{html.escape(hint)}</span>' if hint else ""
    if "enum" in spec:
        opts = "".join(
            f'<option value="{html.escape(str(v))}">{html.escape(YES_NO_NUM.get(v, str(v)) if name == "SeniorCitizen" else str(v))}</option>'
            for v in spec["enum"]
        )
        control = f'<select id="f-{name}" name="{name}" required>{opts}</select>'
    elif name == "TotalCharges":
        control = f'<input id="f-{name}" name="{name}" type="number" min="0" step="0.01" inputmode="decimal">'
    else:
        lo, hi = spec.get("minimum", 0), spec.get("maximum")
        step = "1" if spec.get("type") == "integer" else "0.01"
        control = (f'<input id="f-{name}" name="{name}" type="number" min="{lo}" max="{hi}" '
                   f'step="{step}" required inputmode="decimal">')
    return f'<label for="f-{name}"><span>{html.escape(label)}</span>{control}{hint_html}</label>'


def _form_html() -> str:
    schema = _field_schema()
    parts = []
    for title, names in GROUPS:
        controls = "".join(_control(n, schema[n]) for n in names)
        parts.append(f"<fieldset><legend>{html.escape(title)}</legend><div class=\"grid\">{controls}</div></fieldset>")
    return "".join(parts)


def render_index() -> str:
    """The complete page as a string."""
    schema = _field_schema()
    config = {
        "fields": list(CustomerFeatures.model_fields),
        "numeric": [n for n, s in schema.items() if s.get("type") in ("integer", "number")
                    or "anyOf" in s],
        "presets": PRESETS,
        "maxRows": 1000,
    }
    return (_PAGE.replace("__FORM__", _form_html())
                 .replace("__CONFIG__", json.dumps(config).replace("</", "<\\/")))


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Churn risk checker</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --ink:#1b1f24; --muted:#5b6572; --line:#d9dee5; --accent:#1f5eff;
        --low:#1a7f4b; --mid:#a15c00; --high:#b42318; --lowbg:#e6f4ec; --midbg:#fdf1de; --highbg:#fde8e6; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1318; --card:#171c23; --ink:#e8ecf1; --muted:#9aa5b1; --line:#2b333d; --accent:#6f95ff;
          --low:#5fd08f; --mid:#f0b45a; --high:#ff8a80; --lowbg:#12281d; --midbg:#2e2412; --highbg:#321a18; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
header, main { max-width:980px; margin:0 auto; padding:0 16px; }
header { padding-top:28px; }
h1 { font-size:1.6rem; margin:0 0 4px; }
.sub { color:var(--muted); margin:0 0 16px; }
.status { display:inline-block; font-size:.85rem; padding:2px 10px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }
.status.ok { color:var(--low); border-color:var(--low); }
.status.bad { color:var(--high); border-color:var(--high); }
.notice { background:var(--midbg); border:1px solid var(--mid); color:var(--ink); border-radius:8px; padding:10px 14px; margin:16px 0; font-size:.92rem; }
nav { display:flex; gap:8px; margin:16px 0 0; }
nav button { font:inherit; padding:8px 16px; border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:8px 8px 0 0; cursor:pointer; }
nav button[aria-selected="true"] { border-bottom-color:var(--card); font-weight:600; box-shadow:inset 0 -3px 0 var(--accent); }
section.panel { background:var(--card); border:1px solid var(--line); border-radius:0 8px 8px 8px; padding:16px; margin-bottom:32px; }
fieldset { border:1px solid var(--line); border-radius:8px; margin:0 0 16px; padding:8px 12px 14px; }
legend { padding:0 6px; font-weight:600; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:12px 16px; }
label { display:flex; flex-direction:column; gap:4px; font-size:.92rem; }
label span { font-weight:500; }
.hint { color:var(--muted); font-size:.8rem; }
select, input[type=number], input[type=file] { font:inherit; padding:8px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); width:100%; }
select:focus, input:focus, button:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
button.primary { font:inherit; font-weight:600; padding:10px 20px; border:0; border-radius:8px; background:var(--accent); color:#fff; cursor:pointer; }
button.secondary { font:inherit; padding:8px 14px; border:1px solid var(--line); border-radius:8px; background:var(--card); color:var(--ink); cursor:pointer; }
button:disabled { opacity:.55; cursor:progress; }
.result { margin-top:16px; border-radius:10px; padding:16px; border:1px solid var(--line); }
.result.low { background:var(--lowbg); border-color:var(--low); }
.result.mid { background:var(--midbg); border-color:var(--mid); }
.result.high { background:var(--highbg); border-color:var(--high); }
.score { font-size:2rem; font-weight:700; line-height:1.1; }
.badge { display:inline-block; font-weight:600; padding:2px 10px; border-radius:999px; border:1px solid currentColor; margin-left:8px; font-size:.95rem; vertical-align:middle; }
.low .badge { color:var(--low); } .mid .badge { color:var(--mid); } .high .badge { color:var(--high); }
.small { color:var(--muted); font-size:.85rem; }
.error { color:var(--high); margin-top:12px; white-space:pre-line; }
table { border-collapse:collapse; width:100%; font-size:.9rem; margin-top:12px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
th { position:sticky; top:0; background:var(--card); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.tag { padding:1px 8px; border-radius:999px; font-size:.8rem; border:1px solid currentColor; }
.tag.low { color:var(--low); } .tag.mid { color:var(--mid); } .tag.high { color:var(--high); }
.scroll { max-height:420px; overflow:auto; border:1px solid var(--line); border-radius:8px; margin-top:12px; }
.scroll table { margin:0; }
footer { max-width:980px; margin:0 auto; padding:0 16px 32px; color:var(--muted); font-size:.85rem; }
a { color:var(--accent); }
[hidden] { display:none !important; }
</style>
</head>
<body>
<header>
  <h1>Churn risk checker</h1>
  <p class="sub">Estimate how likely a telecom customer is to leave, one at a time or from a spreadsheet. <span id="status" class="status" role="status">Checking service&hellip;</span></p>
  <div class="notice"><strong>Demo model, not for real decisions.</strong> It was trained on a public telecom dataset (7,043 customers). The score ranks customers against each other; it is <em>not</em> a literal percentage chance, so use it to compare customers, not to quote odds.</div>
</header>
<main>
  <nav role="tablist" aria-label="Mode">
    <button role="tab" id="tab-one" aria-selected="true" aria-controls="panel-one">One customer</button>
    <button role="tab" id="tab-many" aria-selected="false" aria-controls="panel-many">Many customers (CSV)</button>
  </nav>

  <section class="panel" id="panel-one" role="tabpanel" aria-labelledby="tab-one">
    <div class="row" style="margin-bottom:12px">
      <span class="small">Start from an example:</span>
      <button type="button" class="secondary" data-preset="new_flexible">New customer, month-to-month, fiber</button>
      <button type="button" class="secondary" data-preset="long_committed">Long-time customer, two-year contract</button>
    </div>
    <form id="one-form" novalidate>
      __FORM__
      <div class="row"><button class="primary" type="submit" id="one-go">Check risk</button><button class="secondary" type="reset" id="one-reset">Clear</button></div>
    </form>
    <div id="one-result" aria-live="polite"></div>
  </section>

  <section class="panel" id="panel-many" role="tabpanel" aria-labelledby="tab-many" hidden>
    <p>Upload a CSV with one customer per row (up to <span id="max-rows"></span>). Required columns are the customer fields; an optional <code>customerID</code> column is kept so you can match results back.</p>
    <div class="row">
      <button type="button" class="secondary" id="template">Download a template CSV</button>
      <input type="file" id="file" accept=".csv,text/csv" aria-label="CSV file">
      <button class="primary" type="button" id="many-go" disabled>Score customers</button>
    </div>
    <div id="many-msg" aria-live="polite"></div>
    <div id="many-out" hidden>
      <div class="row" style="margin-top:12px"><span id="many-summary"></span><button type="button" class="secondary" id="download">Download results (CSV)</button></div>
      <div class="scroll"><table><thead><tr><th>#</th><th>Customer</th><th>Contract</th><th>Tenure</th><th>Monthly $</th><th>Risk score</th><th>Level</th></tr></thead><tbody id="many-rows"></tbody></table></div>
    </div>
  </section>
</main>
<footer>
  Built for the AI Experiment Lab project &middot; <a href="docs">API documentation</a> &middot; <a href="https://github.com/sudhanshu230230/AI_Experiment_Lab">source</a>.
  The hosted demo runs on a free tier and sleeps when idle, so the first request after a pause can take about a minute.
</footer>
<script>
const CFG = __CONFIG__;
const $ = (id) => document.getElementById(id);
let threshold = 0.5, modelVersion = null;

// ---- helpers ------------------------------------------------------------------------------
function level(score, flagged) {
  if (flagged || score >= threshold) return ["high", "High risk"];
  if (score >= threshold * 0.6) return ["mid", "Medium risk"];
  return ["low", "Low risk"];
}
async function api(path, body, retries) {
  // The free host may be asleep: retry a few times with a visible message.
  let lastErr;
  for (let i = 0; i <= (retries || 0); i++) {
    try {
      const r = await fetch(path, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : undefined);
      let data = null; try { data = await r.json(); } catch (_) {}
      if (r.status === 502 || r.status === 503 || r.status === 504) { lastErr = new Error("The service is waking up."); await new Promise(res => setTimeout(res, 4000)); continue; }
      return {status: r.status, data};
    } catch (e) { lastErr = e; await new Promise(res => setTimeout(res, 4000)); }
  }
  throw lastErr || new Error("Could not reach the service.");
}
function explainError(data, rowOffset) {
  // FastAPI 422: {detail:[{loc:[...], msg}]}
  if (!data || !Array.isArray(data.detail)) return (data && data.detail) || "The service rejected the request.";
  return data.detail.slice(0, 10).map(d => {
    const loc = d.loc || []; const i = loc.indexOf("customers");
    const row = i >= 0 && typeof loc[i + 1] === "number" ? "Row " + (loc[i + 1] + 1 + rowOffset) + ": " : "";
    const field = loc.filter(x => typeof x === "string" && x !== "body" && x !== "customers").join(".");
    return row + (field ? field + " - " : "") + (d.msg || "invalid");
  }).join("\n") + (data.detail.length > 10 ? "\n... and " + (data.detail.length - 10) + " more" : "");
}

// ---- service status -----------------------------------------------------------------------
async function checkStatus() {
  const el = $("status");
  try {
    const {status, data} = await api("health", null, 0);
    if (status === 200 && data.model_loaded) {
      el.textContent = "Service ready"; el.className = "status ok";
      try { const m = await api("model"); if (m.status === 200) { threshold = m.data.threshold || 0.5; modelVersion = m.data.registry_version; el.textContent = "Service ready, model v" + modelVersion; } } catch (_) {}
    } else { el.textContent = "Service up, model not loaded"; el.className = "status bad"; }
  } catch (_) { el.textContent = "Service waking up (about a minute)"; el.className = "status"; setTimeout(checkStatus, 5000); }
}

// ---- tabs ---------------------------------------------------------------------------------
for (const [tab, panel] of [["tab-one", "panel-one"], ["tab-many", "panel-many"]]) {
  $(tab).addEventListener("click", () => {
    for (const [t, p] of [["tab-one", "panel-one"], ["tab-many", "panel-many"]]) { $(t).setAttribute("aria-selected", String(t === tab)); $(p).hidden = p !== panel; }
  });
}

// ---- one customer -------------------------------------------------------------------------
function readForm() {
  const out = {};
  for (const f of CFG.fields) {
    const el = $("f-" + f); let v = el.value;
    if (CFG.numeric.includes(f)) { v = v === "" ? (f === "TotalCharges" ? null : NaN) : Number(v); }
    else if (f === "SeniorCitizen") { v = Number(v); }
    out[f] = v;
  }
  return out;
}
function fillForm(obj) { for (const f of CFG.fields) { const el = $("f-" + f); const v = obj[f]; el.value = v === null || v === undefined ? "" : String(v); } }
document.querySelectorAll("[data-preset]").forEach(b => b.addEventListener("click", () => { fillForm(CFG.presets[b.dataset.preset]); $("one-result").replaceChildren(); }));

$("one-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = $("one-result"); out.replaceChildren();
  const form = $("one-form");
  if (!form.reportValidity()) return;
  const btn = $("one-go"); btn.disabled = true; btn.textContent = "Checking...";
  try {
    const {status, data} = await api("predict", readForm(), 6);
    if (status !== 200) { const p = document.createElement("div"); p.className = "error"; p.textContent = explainError(data, 0); out.append(p); return; }
    const [cls, text] = level(data.churn_probability, data.prediction === "Yes");
    const box = document.createElement("div"); box.className = "result " + cls;
    const score = document.createElement("div"); score.className = "score";
    score.textContent = "Risk score " + Math.round(data.churn_probability * 100) + " / 100";
    const badge = document.createElement("span"); badge.className = "badge"; badge.textContent = text; score.append(badge);
    const note = document.createElement("p"); note.className = "small";
    note.textContent = (data.prediction === "Yes" ? "The model flags this customer as likely to leave" : "The model does not flag this customer as likely to leave")
      + " (flag threshold " + Math.round(data.threshold * 100) + "). Scores rank customers against each other and are not literal percentages. Model version " + data.model_version + ".";
    box.append(score, note); out.append(box);
  } catch (e) {
    const p = document.createElement("div"); p.className = "error"; p.textContent = "Could not reach the service: " + e.message; out.append(p);
  } finally { btn.disabled = false; btn.textContent = "Check risk"; }
});
$("one-reset").addEventListener("click", () => $("one-result").replaceChildren());

// ---- many customers (CSV) -----------------------------------------------------------------
$("max-rows").textContent = CFG.maxRows.toLocaleString();
let parsed = null, results = null;

function parseCSV(text) {
  const rows = []; let row = [], cell = "", q = false;
  text = text.replace(/^﻿/, "");
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(cell); cell = ""; if (row.length > 1 || row[0] !== "") rows.push(row); row = []; }
    else cell += c;
  }
  if (cell !== "" || row.length) { row.push(cell); if (row.length > 1 || row[0] !== "") rows.push(row); }
  return rows;
}
function csvEscape(v) { v = v === null || v === undefined ? "" : String(v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function setMsg(text, isError) { const m = $("many-msg"); m.className = isError ? "error" : "small"; m.textContent = text; }

$("template").addEventListener("click", () => {
  const ex = CFG.presets.new_flexible;
  const header = ["customerID", ...CFG.fields];
  const line = ["EXAMPLE-001", ...CFG.fields.map(f => ex[f])];
  const blob = new Blob([header.map(csvEscape).join(",") + "\n" + line.map(csvEscape).join(",") + "\n"], {type: "text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "customers_template.csv"; a.click(); URL.revokeObjectURL(a.href);
});

$("file").addEventListener("change", async () => {
  parsed = null; results = null; $("many-go").disabled = true; $("many-out").hidden = true; setMsg("", false);
  const f = $("file").files[0]; if (!f) return;
  if (f.size > 5 * 1024 * 1024) { setMsg("That file is over 5 MB. Split it into smaller files.", true); return; }
  const rows = parseCSV(await f.text());
  if (rows.length < 2) { setMsg("The file needs a header row and at least one customer.", true); return; }
  const header = rows[0].map(h => h.trim());
  const missing = CFG.fields.filter(c => !header.includes(c));
  if (missing.length) { setMsg("Missing column(s): " + missing.join(", ") + ". Use the template for the exact names.", true); return; }
  const idCol = header.find(h => h.toLowerCase() === "customerid");
  const body = rows.slice(1);
  if (body.length > CFG.maxRows) { setMsg("The file has " + body.length.toLocaleString() + " customers; the limit is " + CFG.maxRows.toLocaleString() + " per upload.", true); return; }
  const customers = [], ids = [];
  body.forEach((r, i) => {
    const o = {};
    for (const c of CFG.fields) {
      let v = (r[header.indexOf(c)] ?? "").trim();
      if (CFG.numeric.includes(c)) v = v === "" ? (c === "TotalCharges" ? null : NaN) : Number(v);
      else if (c === "SeniorCitizen") v = Number(v);
      o[c] = v;
    }
    customers.push(o); ids.push(idCol ? (r[header.indexOf(idCol)] ?? "").trim() : String(i + 1));
  });
  parsed = {customers, ids};
  setMsg(customers.length.toLocaleString() + " customers ready.", false);
  $("many-go").disabled = false;
});

$("many-go").addEventListener("click", async () => {
  if (!parsed) return;
  const btn = $("many-go"); btn.disabled = true; btn.textContent = "Scoring...";
  try {
    const {status, data} = await api("predict/batch", {customers: parsed.customers}, 6);
    if (status !== 200) { setMsg(explainError(data, 0), true); return; }
    results = data.predictions.map((p, i) => ({i: i + 1, id: parsed.ids[i], c: parsed.customers[i], score: p.churn_probability, flag: p.prediction === "Yes"}));
    renderResults(); setMsg("", false);
  } catch (e) { setMsg("Could not reach the service: " + e.message, true); }
  finally { btn.disabled = false; btn.textContent = "Score customers"; }
});

function renderResults() {
  const rows = [...results].sort((a, b) => b.score - a.score);
  const body = $("many-rows"); body.replaceChildren();
  let high = 0;
  for (const r of rows) {
    const [cls, text] = level(r.score, r.flag); if (r.flag) high++;
    const tr = document.createElement("tr");
    const cells = [r.i, r.id, r.c.Contract, r.c.tenure, Number(r.c.MonthlyCharges).toFixed(2), Math.round(r.score * 100)];
    cells.forEach((v, k) => { const td = document.createElement("td"); if (k >= 3) td.className = "num"; td.textContent = v; tr.append(td); });
    const td = document.createElement("td"); const tag = document.createElement("span"); tag.className = "tag " + cls; tag.textContent = text; td.append(tag); tr.append(td);
    body.append(tr);
  }
  $("many-summary").textContent = results.length.toLocaleString() + " scored; " + high.toLocaleString() + " flagged as likely to leave. Highest risk first.";
  $("many-out").hidden = false;
}
$("download").addEventListener("click", () => {
  if (!results) return;
  const header = ["customerID", ...CFG.fields, "risk_score", "flagged_likely_to_leave", "risk_level"];
  const lines = [header.map(csvEscape).join(",")];
  for (const r of results) {
    const [, text] = level(r.score, r.flag);
    lines.push([r.id, ...CFG.fields.map(f => r.c[f]), r.score.toFixed(4), r.flag ? "Yes" : "No", text].map(csvEscape).join(","));
  }
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/csv"}));
  a.download = "churn_risk_results.csv"; a.click(); URL.revokeObjectURL(a.href);
});

fillForm(CFG.presets.new_flexible);
checkStatus();
</script>
</body>
</html>
"""
