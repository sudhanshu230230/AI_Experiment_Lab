"""The web page served at ``/app``: a form for one row and a CSV upload for many.

The page is generated from a "UI spec" (``ui_spec``). For a model exported by ``automl`` the spec
comes from the schema stored in the model's metadata, so any dataset gets a matching form,
CSV template and result display. The Telco demo (an export without a schema) keeps its
hand-written labels, groups and examples.

Plain HTML and a little JavaScript: no build step, no extra dependency.
"""

from __future__ import annotations

import html
import json
from typing import Any

from ai_experiment_lab.automl.schema import Feature, element_ids
from ai_experiment_lab.serving import theme
from ai_experiment_lab.serving.contract import MAX_BATCH, Contract
from ai_experiment_lab.serving.schemas import CustomerFeatures

# (label, help text) shown to people instead of the raw column names (Telco demo only).
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


def _pretty(name: str) -> str:
    text = name.replace("_", " ").strip()
    if text and text == text.lower():
        text = text[0].upper() + text[1:]
    return text


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.4g}"


# ---- spec building -------------------------------------------------------------------------------

def _legacy_spec() -> dict[str, Any]:
    props = CustomerFeatures.model_json_schema()["properties"]
    names = list(CustomerFeatures.model_fields)
    ids = element_ids(names)
    fields = []
    for name, fid in zip(names, ids, strict=True):
        spec = props[name]
        label, hint = LABELS[name]
        f: dict[str, Any] = {"name": name, "id": fid, "label": label, "hint": hint,
                             "required": name != "TotalCharges"}
        if "enum" in spec:
            numeric = all(isinstance(v, int) for v in spec["enum"])
            f.update(input="select", send="number" if numeric else "string",
                     options=[{"value": str(v), "label": YES_NO_NUM.get(v, str(v))
                               if name == "SeniorCitizen" else str(v)} for v in spec["enum"]])
        else:
            lo, hi = spec.get("minimum", 0), spec.get("maximum")
            f.update(input="number", send="number", nullable=name == "TotalCharges",
                     min=None if name == "TotalCharges" else lo,
                     max=hi, step="1" if spec.get("type") == "integer" else "0.01")
        fields.append(f)
    return {
        "legacy": True, "title": "Churn risk checker", "task": "binary", "target": "Churn",
        "classes": ["No", "Yes"], "positive_label": "Yes", "negative_label": "No",
        "intro": "Estimate how likely a telecom customer is to leave, one at a time or from a spreadsheet.",
        "notice": ("<strong>Demo model, not for real decisions.</strong> It was trained on a public "
                   "telecom dataset (7,043 customers). The score ranks customers against each "
                   "other; it is <em>not</em> a literal percentage chance, so use it to compare "
                   "customers, not to quote odds."),
        "noun": "risk", "one_title": "One customer", "many_title": "Many customers (CSV)",
        "summary": ["Contract", "tenure", "MonthlyCharges"],
        "groups": [{"title": t, "names": n} for t, n in GROUPS],
        "presets": [
            {"label": "New customer, month-to-month, fiber", "values": PRESETS["new_flexible"]},
            {"label": "Long-time customer, two-year contract", "values": PRESETS["long_committed"]},
        ],
        "fields": fields,
    }


def _generic_field(f: Feature, fid: str) -> dict[str, Any]:
    out: dict[str, Any] = {"name": f.name, "id": fid, "label": _pretty(f.name),
                           "required": not (f.nullable or f.numeric_text), "hint": ""}
    if f.kind == "categorical":
        if f.allowed:
            out.update(input="select", send="string",
                       options=[{"value": v, "label": v} for v in f.allowed])
        else:
            out.update(input="text", send="string")
        if f.nullable:
            out["hint"] = "Optional"
        return out
    if f.choices:
        out.update(input="select", send="number",
                   options=[{"value": str(v), "label": str(v)} for v in f.choices])
        if f.nullable:
            out["hint"] = "Optional"
        return out
    lo, hi = f.min, f.max
    out.update(input="number", send="number", nullable=not out["required"],
               step="1" if f.integer else "any")
    if lo is not None and hi is not None:
        out["hint"] = f"Training data: {_fmt(lo)} to {_fmt(hi)}"
    if not out["required"]:
        out["hint"] = (out["hint"] + ". " if out["hint"] else "") + "Leave blank if unknown"
    return out


def _generic_spec(contract: Contract) -> dict[str, Any]:
    schema = contract.schema
    assert schema is not None
    ids = element_ids(schema.names)
    fields = [_generic_field(f, i) for f, i in zip(schema.features, ids, strict=True)]
    if len(fields) > 12:
        nums = [f["name"] for f in fields if f["input"] == "number"]
        cats = [f["name"] for f in fields if f["input"] != "number"]
        groups = [g for g in ({"title": "Numbers", "names": nums},
                              {"title": "Categories", "names": cats}) if g["names"]]
    else:
        groups = [{"title": "Details", "names": schema.names}]
    task, target = schema.task, schema.target
    if task == "binary":
        pos = schema.positive_label
        notice = (f"This model estimates the score for <strong>{html.escape(str(target))} = "
                  f"{html.escape(str(pos))}</strong>. The score ranks rows against each other; it is "
                  "<em>not</em> a calibrated probability, so compare rows rather than quoting odds.")
    elif task == "multiclass":
        notice = (f"This model predicts <strong>{html.escape(str(target))}</strong> as one of "
                  f"{len(schema.classes)} classes and shows how confident it is in each.")
    else:
        notice = (f"This model predicts the number <strong>{html.escape(str(target))}</strong>. "
                  "It does not give a confidence range, so treat the value as an estimate.")
    notice += (" It was trained on one dataset; check its results against your own data "
               "before relying on them.")
    example = schema.example()
    return {
        "legacy": False, "title": contract.title, "task": task, "target": target,
        "classes": schema.classes, "positive_label": schema.positive_label,
        "negative_label": schema.negative_label,
        "intro": f"Enter the details of one row, or upload a CSV, to predict {target}.",
        "notice": notice, "noun": "score", "one_title": "One row",
        "many_title": "Many rows (CSV)",
        "summary": schema.names[:3],
        "groups": groups,
        "presets": [{"label": "Fill with a typical example", "values": example}],
        "fields": fields,
    }


def ui_spec(contract: Contract) -> dict[str, Any]:
    """The description of the page and of ``/schema``: fields, labels, task and wording."""
    spec = _legacy_spec() if contract.legacy else _generic_spec(contract)
    spec["max_rows"] = MAX_BATCH
    return spec


# ---- html ---------------------------------------------------------------------------------------

def _control(f: dict[str, Any]) -> str:
    hint = f'<span class="hint">{html.escape(f["hint"])}</span>' if f.get("hint") else ""
    fid, name = f["id"], f["name"]
    req = " required" if f["required"] else ""
    if f["input"] == "select":
        opts = "".join(f'<option value="{html.escape(o["value"])}">{html.escape(o["label"])}</option>'
                       for o in f["options"])
        ctl = f'<select id="f-{fid}" name="{html.escape(name)}"{req}>{opts}</select>'
        if not f["required"]:
            ctl = ctl.replace("<select", "<select", 1).replace(
                ">", '><option value="">(not set)</option>', 1)
    elif f["input"] == "text":
        ctl = (f'<input id="f-{fid}" name="{html.escape(name)}" type="text" maxlength="200"{req}>')
    else:
        lo = f' min="{f["min"]}"' if f.get("min") is not None else ""
        hi = f' max="{f["max"]}"' if f.get("max") is not None else ""
        ctl = (f'<input id="f-{fid}" name="{html.escape(name)}" type="number"{lo}{hi} '
               f'step="{f.get("step", "any")}" inputmode="decimal"{req}>')
    return f'<label for="f-{fid}"><span>{html.escape(f["label"])}</span>{ctl}{hint}</label>'


def _form_html(spec: dict[str, Any]) -> str:
    by_name = {f["name"]: f for f in spec["fields"]}
    parts = []
    for g in spec["groups"]:
        controls = "".join(_control(by_name[n]) for n in g["names"])
        parts.append(f'<fieldset><legend>{html.escape(g["title"])}</legend>'
                     f'<div class="grid">{controls}</div></fieldset>')
    return "".join(parts)


def _script_json(obj: Any) -> str:
    """JSON that is safe inside a ``<script>`` block: no ``<``, ``>`` or ``&`` can survive, so a
    column name such as ``</script><script>...`` cannot close the block or open a new one."""
    return (json.dumps(obj).replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))


def render_index(spec: dict[str, Any] | None = None) -> str:
    """The complete page as a string. Without a spec it is the Telco demo page."""
    if spec is None:
        spec = ui_spec(Contract(True, "binary", list(CustomerFeatures.model_fields),
                                CustomerFeatures, CustomerFeatures))
    title = html.escape(spec["title"])
    page = (_PAGE.replace("__THEME_CSS__", theme.CSS).replace("__FONTS__", theme.FONTS)
                 .replace("__TOPBAR__", theme.topbar("predict"))
                 .replace("__FORM__", _form_html(spec))
                 .replace("__TITLE__", title)
                 .replace("__INTRO__", html.escape(spec["intro"]))
                 .replace("__NOTICE__", spec["notice"])
                 .replace("__ONE_TITLE__", html.escape(spec["one_title"]))
                 .replace("__MANY_TITLE__", html.escape(spec["many_title"]))
                 .replace("__CONFIG__", _script_json(spec)))
    return page


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
__FONTS__
<style>
__THEME_CSS__
/* predict page */
.tabs { display:inline-flex; gap:4px; padding:4px; background:var(--sunk); border:1px solid var(--line); border-radius:999px; margin-bottom:22px; }
.tabs button { font:500 .95rem var(--text); padding:9px 20px; border:0; border-radius:999px; background:transparent; color:var(--muted); transition:background .15s,color .15s; }
.tabs button:hover { color:var(--ink); }
.tabs button[aria-selected="true"] { background:var(--reagent); color:var(--reagent-ink); font-weight:600; }
#presets { margin-bottom:8px; }
#presets button { font-size:.88rem; }
fieldset { border:0; margin:0; padding:26px 0 8px; min-width:0; position:relative; }
fieldset::before { content:""; position:absolute; top:0; left:0; right:0; height:1px; background:var(--line); }
legend { padding:0; margin-bottom:12px; font:700 1.1rem var(--display); letter-spacing:-.01em; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:16px 20px; padding-bottom:12px; }
#one-form .row { padding-top:12px; border-top:1px solid var(--line); }
.result { margin-top:24px; border-radius:16px; padding:clamp(16px,3vw,28px); background:var(--sunk); border:1.5px solid var(--line); }
.result.low { background:var(--lowbg); border-color:var(--low); }
.result.mid { background:var(--midbg); border-color:var(--mid); }
.result.high { background:var(--highbg); border-color:var(--high); }
.score { font:800 clamp(1.6rem,4vw,2.4rem)/1.1 var(--display); letter-spacing:-.02em; display:flex; flex-wrap:wrap; align-items:center; gap:10px 14px; }
.badge { font:600 .9rem var(--text); padding:4px 14px; border-radius:999px; background:var(--ink); color:var(--glass); letter-spacing:0; }
.result .small { margin:14px 0 0; max-width:70ch; }
.bars { margin-top:16px; display:grid; gap:8px; }
.bars div { display:grid; grid-template-columns:minmax(80px,160px) 1fr 44px; gap:12px; align-items:center; font-size:.9rem; }
.bars i { display:block; height:10px; border-radius:5px; background:var(--reagent); }
.bars em { font-style:normal; text-align:right; color:var(--muted); }
.drop { position:relative; display:flex; flex-direction:column; align-items:center; gap:4px; padding:30px 16px; text-align:center; border:2px dashed var(--line); border-radius:16px; background:var(--sunk); transition:border-color .15s, background .15s; cursor:pointer; margin:16px 0 12px; }
.drop:hover, .drop.over { border-color:var(--reagent); background:color-mix(in srgb, var(--reagent) 9%, var(--sunk)); }
.drop strong { font:700 1.1rem var(--display); }
.drop input { position:absolute; inset:0; width:100%; height:100%; opacity:0; cursor:pointer; padding:0; }
.drop:focus-within { outline:3px solid var(--focus); outline-offset:2px; }
.steps { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
.labcta { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px 24px; padding:18px 22px; border-radius:16px; background:var(--deep); color:var(--deep-ink); }
.labcta strong { font:700 1.05rem var(--display); display:block; }
.labcta span { color:var(--deep-muted); font-size:.92rem; }
.labcta a { background:var(--deep-ink); color:var(--deep); }
footer { max-width:1040px; margin:0 auto; padding:0 20px 40px; color:var(--muted); font-size:.85rem; }
#many-msg .primary { margin:4px 12px 0 0; }
</style>
</head>
<body>
__TOPBAR__
<div class="hero"><div class="in">
  <h1>__TITLE__</h1>
  <p>__INTRO__</p>
  <span id="status" class="status" role="status">Checking service&hellip;</span>
</div></div>
<main class="wrap">
  <div class="sheet">
  <div class="notice">__NOTICE__</div>
  <div class="tabs" role="tablist" aria-label="Mode">
    <button role="tab" id="tab-one" aria-selected="true" aria-controls="panel-one">__ONE_TITLE__</button>
    <button role="tab" id="tab-many" aria-selected="false" aria-controls="panel-many">__MANY_TITLE__</button>
  </div>

  <section id="panel-one" role="tabpanel" aria-labelledby="tab-one">
    <div class="row" id="presets"><span class="small">Start from an example:</span></div>
    <form id="one-form" novalidate>
      __FORM__
      <div class="row"><button class="primary" type="submit" id="one-go">Predict</button><button class="secondary" type="reset" id="one-reset">Clear</button></div>
    </form>
    <div id="one-result" aria-live="polite"></div>
  </section>

  <section id="panel-many" role="tabpanel" aria-labelledby="tab-many" hidden>
    <p style="margin-top:0;max-width:68ch">Upload a CSV with one row per line (up to <span id="max-rows"></span>). The columns must have the same names as the form fields; use the template for the exact names. Extra columns, such as an ID, are kept in the results and not sent to the model.</p>
    <button type="button" class="secondary" id="template">Download a template CSV</button>
    <label class="drop" id="drop"><strong id="drop-title">Choose a CSV file</strong><span class="small" id="drop-sub">or drop it here</span>
      <input type="file" id="file" accept=".csv,text/csv" aria-label="CSV file"></label>
    <div class="row"><button class="primary" type="button" id="many-go" disabled>Predict all rows</button></div>
    <div id="many-msg" aria-live="polite"></div>
    <div id="many-out" hidden>
      <div class="row" style="margin:20px 0 12px;justify-content:space-between"><span id="many-summary"></span><button type="button" class="secondary" id="download">Download results (CSV)</button></div>
      <div class="scroll" style="max-height:460px"><table><thead id="many-head"></thead><tbody id="many-rows"></tbody></table></div>
    </div>
  </section>
  </div>

  <div class="labcta" id="lab-link" hidden>
    <div><strong>Have your own table of data?</strong><span>Upload any CSV and find out which algorithm to use, and with what settings.</span></div>
    <a class="btn" href="lab">Open the advisor</a>
  </div>
</main>
<footer>
  Built with the AI Experiment Lab platform &middot; <a href="docs">API documentation</a> &middot; <a href="https://github.com/sudhanshu230230/AI_Experiment_Lab">source</a>.
  The hosted demo runs on a free tier and sleeps when idle, so the first request after a pause can take about a minute.
</footer>
<script>
const CFG = __CONFIG__;
const $ = (id) => document.getElementById(id);
const FIELDS = CFG.fields;
let threshold = 0.5;

// ---- helpers ------------------------------------------------------------------------------
function pct(x) { return Math.round(x * 100); }
function fmtNum(v) { return Math.abs(v) >= 1000 ? v.toLocaleString(undefined, {maximumFractionDigits: 2}) : String(Math.round(v * 10000) / 10000); }
function level(score, flagged) {
  if (flagged || score >= threshold) return ["high", "High " + CFG.noun];
  if (score >= threshold * 0.6) return ["mid", "Medium " + CFG.noun];
  return ["low", "Low " + CFG.noun];
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
function explainError(data) {
  // FastAPI 422: {detail:[{loc:[...], msg}]}
  if (!data || !Array.isArray(data.detail)) return (data && data.detail) || "The service rejected the request.";
  return data.detail.slice(0, 10).map(d => {
    const loc = d.loc || []; const i = loc.indexOf("customers");
    const row = i >= 0 && typeof loc[i + 1] === "number" ? "Row " + (loc[i + 1] + 1) + ": " : "";
    const field = loc.filter(x => typeof x === "string" && x !== "body" && x !== "customers").join(".");
    return row + (field ? field + " - " : "") + (d.msg || "invalid");
  }).join("\n") + (data.detail.length > 10 ? "\n... and " + (data.detail.length - 10) + " more" : "");
}
function convert(f, raw) {
  // raw is a string from the form or CSV; returns the JSON value for the API.
  const v = (raw === null || raw === undefined) ? "" : String(raw).trim();
  if (f.send === "number") { return v === "" ? (f.required ? NaN : null) : Number(v); }
  if (v === "") return f.required ? "" : null;
  return v;
}

fetch("lab/limits").then(r => { if (r.ok) { labOn = true; $("lab-link").hidden = false; } }).catch(() => {});

// ---- service status -----------------------------------------------------------------------
async function checkStatus() {
  const el = $("status");
  try {
    const {status, data} = await api("health", null, 0);
    if (status === 200 && data.model_loaded) {
      el.textContent = "Service ready"; el.className = "status ok";
      try { const m = await api("model"); if (m.status === 200) { threshold = m.data.threshold || 0.5; el.textContent = "Service ready, model" + (m.data.registry_version != null ? " v" + m.data.registry_version : ""); } } catch (_) {}
    } else { el.textContent = "Service up, model not loaded"; el.className = "status bad"; }
  } catch (_) { el.textContent = "Service waking up (about a minute)"; el.className = "status"; setTimeout(checkStatus, 5000); }
}

// ---- tabs ---------------------------------------------------------------------------------
for (const [tab, panel] of [["tab-one", "panel-one"], ["tab-many", "panel-many"]]) {
  $(tab).addEventListener("click", () => {
    for (const [t, p] of [["tab-one", "panel-one"], ["tab-many", "panel-many"]]) { $(t).setAttribute("aria-selected", String(t === tab)); $(p).hidden = p !== panel; }
  });
}

// ---- one row ------------------------------------------------------------------------------
function readForm() {
  const out = {};
  for (const f of FIELDS) out[f.name] = convert(f, $("f-" + f.id).value);
  return out;
}
function fillForm(obj) {
  for (const f of FIELDS) { const v = obj[f.name]; $("f-" + f.id).value = v === null || v === undefined ? "" : String(v); }
}
const presetBox = $("presets");
CFG.presets.forEach((p, i) => {
  const b = document.createElement("button"); b.type = "button"; b.className = "secondary"; b.textContent = p.label;
  b.addEventListener("click", () => { fillForm(p.values); $("one-result").replaceChildren(); });
  presetBox.append(b);
});

function describe(p) {
  // One prediction as {cls, headline, badge, note} for the result card.
  if (CFG.task === "regression") {
    return {cls: "mid", headline: "Predicted " + CFG.target + ": " + fmtNum(p.prediction), badge: null,
            note: "A single estimate with no confidence range. Model version " + (p.model_version ?? "n/a") + "."};
  }
  if (CFG.task === "multiclass") {
    const probs = Object.entries(p.probabilities || {}).sort((a, b) => b[1] - a[1]);
    const top = probs[0] ? pct(probs[0][1]) : null;
    return {cls: top !== null && top >= 70 ? "low" : "mid", headline: "Predicted " + CFG.target + ": " + p.prediction,
            badge: top !== null ? top + "% confident" : null, bars: probs.slice(0, 6),
            note: "Class probabilities: " + probs.slice(0, 6).map(([k, v]) => k + " " + pct(v) + "%").join(", ") + (probs.length > 6 ? ", ..." : "") + ". Model version " + (p.model_version ?? "n/a") + "."};
  }
  const prob = p.probability ?? p.churn_probability;
  const flagged = p.prediction === CFG.positive_label;
  const [cls, text] = level(prob, flagged);
  const word = CFG.legacy ? "Risk score" : "Score for " + CFG.target + " = " + CFG.positive_label;
  return {cls, headline: word + " " + pct(prob) + " / 100", badge: text, scale: {value: prob, threshold: p.threshold},
          note: (flagged ? "The model predicts " + CFG.positive_label : "The model predicts " + CFG.negative_label)
            + " (threshold " + pct(p.threshold) + "). Scores rank rows against each other and are not literal percentages. Model version " + (p.model_version ?? "n/a") + "."};
}

$("one-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = $("one-result"); out.replaceChildren();
  const form = $("one-form");
  if (!form.reportValidity()) return;
  const btn = $("one-go"); btn.disabled = true; btn.textContent = "Working...";
  try {
    const {status, data} = await api("predict", readForm(), 6);
    if (status !== 200) { const p = document.createElement("div"); p.className = "error"; p.textContent = explainError(data); out.append(p); return; }
    const d = describe(data);
    const box = document.createElement("div"); box.className = "result " + d.cls;
    const score = document.createElement("div"); score.className = "score"; score.textContent = d.headline;
    if (d.badge) { const badge = document.createElement("span"); badge.className = "badge"; badge.textContent = d.badge; score.append(badge); }
    const note = document.createElement("p"); note.className = "small"; note.textContent = d.note;
    box.append(score);
    if (d.scale) {
      const sc = document.createElement("div"); sc.className = "scale"; sc.setAttribute("role", "img");
      sc.setAttribute("aria-label", "Score " + pct(d.scale.value) + " on a scale of 0 to 100");
      const tick = document.createElement("div"); tick.className = "tick"; tick.style.left = pct(d.scale.threshold) + "%";
      const tl = document.createElement("b"); tl.textContent = "decision line " + pct(d.scale.threshold); tick.append(tl);
      const pin = document.createElement("div"); pin.className = "pin"; pin.style.left = "0%";
      sc.append(tick, pin);
      const ends = document.createElement("div"); ends.className = "scale-ends";
      for (const t of ["0 least likely", "100 most likely"]) { const e = document.createElement("span"); e.textContent = t; ends.append(e); }
      box.append(sc, ends);
      requestAnimationFrame(() => requestAnimationFrame(() => { pin.style.left = pct(d.scale.value) + "%"; }));
    }
    if (d.bars) {
      const bars = document.createElement("div"); bars.className = "bars";
      for (const [k, v] of d.bars) {
        const r = document.createElement("div"); const n = document.createElement("span"); n.textContent = k;
        const i = document.createElement("i"); i.style.width = Math.max(2, pct(v)) + "%"; const e = document.createElement("em"); e.textContent = pct(v) + "%";
        const track = document.createElement("span"); track.append(i); r.append(n, track, e); bars.append(r);
      }
      box.append(bars);
    }
    box.append(note); out.append(box);
    box.scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (e) {
    const p = document.createElement("div"); p.className = "error"; p.textContent = "Could not reach the service: " + e.message; out.append(p);
  } finally { btn.disabled = false; btn.textContent = "Predict"; }
});
$("one-reset").addEventListener("click", () => $("one-result").replaceChildren());

// ---- many rows (CSV) ----------------------------------------------------------------------
$("max-rows").textContent = CFG.max_rows.toLocaleString();
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
let labOn = false;
function setMsg(text, isError) { const m = $("many-msg"); m.className = isError ? "error" : "small"; m.textContent = text; }
function offerAdvisor(file) {
  // A file of some other kind of data is not a dead end: hand it to the model advisor.
  if (!labOn) return;
  const m = $("many-msg");
  const p = document.createElement("p"); p.className = "small";
  p.textContent = "Want a model for your own data instead? The advisor tries several algorithms on this file and tells you which to use and with what settings.";
  const b = document.createElement("button"); b.type = "button"; b.className = "primary"; b.textContent = "Find the best algorithm for this file";
  const err = document.createElement("span"); err.className = "error";
  b.addEventListener("click", async () => {
    b.disabled = true; err.textContent = "";
    try {
      const r = await fetch("lab/analyze", {method: "POST", headers: {"Content-Type": "text/csv"}, body: file});
      const body = await r.json().catch(() => null);
      if (!r.ok) { const d = body && body.detail; throw new Error(typeof d === "string" ? d : "The advisor could not read this file."); }
      window.location.href = "lab#upload=" + encodeURIComponent(body.upload_id) + "&name=" + encodeURIComponent(file.name);
    } catch (e) { err.textContent = e.message; b.disabled = false; }
  });
  m.append(p, b, err);
}
const NAMES = FIELDS.map(f => f.name);
const ID_RE = /^(customer_?id|id|row_?id|index|record_?id)$/i;

$("template").addEventListener("click", () => {
  const ex = CFG.presets[0].values;
  const header = ["id", ...NAMES];
  const line = ["EXAMPLE-001", ...NAMES.map(n => ex[n])];
  const blob = new Blob([header.map(csvEscape).join(",") + "\n" + line.map(csvEscape).join(",") + "\n"], {type: "text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "template.csv"; a.click(); URL.revokeObjectURL(a.href);
});

for (const ev of ["dragenter", "dragover"]) $("drop").addEventListener(ev, () => $("drop").classList.add("over"));
for (const ev of ["dragleave", "drop"]) $("drop").addEventListener(ev, () => $("drop").classList.remove("over"));
$("file").addEventListener("change", async () => {
  const picked = $("file").files[0];
  $("drop-title").textContent = picked ? picked.name : "Choose a CSV file";
  $("drop-sub").textContent = picked ? (picked.size / 1024 < 1024 ? Math.round(picked.size / 1024) + " KB" : (picked.size / 1048576).toFixed(1) + " MB") + ", click to choose another" : "or drop it here";
  parsed = null; results = null; $("many-go").disabled = true; $("many-out").hidden = true; setMsg("", false);
  const f = $("file").files[0]; if (!f) return;
  if (f.size > 5 * 1024 * 1024) { setMsg("That file is over 5 MB. Split it into smaller files.", true); return; }
  const rows = parseCSV(await f.text());
  if (rows.length < 2) { setMsg("The file needs a header row and at least one data row.", true); return; }
  const header = rows[0].map(h => h.trim());
  const missing = NAMES.filter(c => !header.includes(c));
  if (missing.length) {
    const mostly = missing.length > NAMES.length / 2;
    setMsg((mostly ? "This file does not look like the data this page scores: " + missing.length + " of the " + NAMES.length + " expected columns are missing (" : "Missing column(s): ") +
      missing.slice(0, 6).join(", ") + (missing.length > 6 ? ", ..." : "") + (mostly ? "). " : ". ") +
      (mostly ? "This page only predicts with the model it was trained on." : "Use the template for the exact names."), true);
    offerAdvisor(f);
    return;
  }
  const idCol = header.find(h => ID_RE.test(h) && !NAMES.includes(h));
  const body = rows.slice(1);
  if (body.length > CFG.max_rows) { setMsg("The file has " + body.length.toLocaleString() + " rows; the limit is " + CFG.max_rows.toLocaleString() + " per upload.", true); return; }
  const customers = [], ids = [];
  body.forEach((r, i) => {
    const o = {};
    for (const f of FIELDS) o[f.name] = convert(f, r[header.indexOf(f.name)]);
    customers.push(o); ids.push(idCol ? (r[header.indexOf(idCol)] ?? "").trim() : String(i + 1));
  });
  parsed = {customers, ids};
  setMsg(customers.length.toLocaleString() + " rows ready.", false);
  $("many-go").disabled = false;
});

function resultCell(r) {
  if (CFG.task === "regression") return [fmtNum(r.p.prediction), null];
  if (CFG.task === "multiclass") { const pr = Object.values(r.p.probabilities || {}); return [String(r.p.prediction), pr.length ? pct(Math.max(...pr)) + "%" : ""]; }
  const prob = r.p.probability ?? r.p.churn_probability;
  const [cls, text] = level(prob, r.p.prediction === CFG.positive_label);
  return [String(pct(prob)), [cls, text]];
}
function sortKey(r) {
  if (CFG.task === "regression") return Number(r.p.prediction);
  if (CFG.task === "multiclass") return Math.max(...Object.values(r.p.probabilities || {0: 0}));
  return r.p.probability ?? r.p.churn_probability;
}

$("many-go").addEventListener("click", async () => {
  if (!parsed) return;
  const btn = $("many-go"); btn.disabled = true; btn.textContent = "Working...";
  try {
    const {status, data} = await api("predict/batch", {customers: parsed.customers}, 6);
    if (status !== 200) { setMsg(explainError(data), true); return; }
    results = data.predictions.map((p, i) => ({i: i + 1, id: parsed.ids[i], c: parsed.customers[i], p}));
    renderResults(); setMsg("", false);
  } catch (e) { setMsg("Could not reach the service: " + e.message, true); }
  finally { btn.disabled = false; btn.textContent = "Predict all rows"; }
});

function renderResults() {
  const rows = [...results].sort((a, b) => sortKey(b) - sortKey(a));
  const head = $("many-head"); head.replaceChildren();
  const tr0 = document.createElement("tr");
  const labelFor = (n) => (FIELDS.find(f => f.name === n) || {label: n}).label;
  const resHead = CFG.task === "regression" ? ["Predicted " + CFG.target] : CFG.task === "multiclass" ? ["Predicted class", "Confidence"] : [CFG.legacy ? "Risk score" : "Score", "Level"];
  ["#", "Row", ...CFG.summary.map(labelFor), ...resHead].forEach(t => { const th = document.createElement("th"); th.textContent = t; tr0.append(th); });
  head.append(tr0);
  const body = $("many-rows"); body.replaceChildren();
  let flagged = 0;
  for (const r of rows) {
    const tr = document.createElement("tr");
    const [main, extra] = resultCell(r);
    const lead = [r.i, r.id, ...CFG.summary.map(n => r.c[n] === null ? "" : r.c[n])];
    lead.forEach((v, k) => { const td = document.createElement("td"); td.textContent = (typeof v === "number" && k >= 2) ? fmtNum(v) : v; if (typeof v === "number" && k >= 2) td.className = "num"; tr.append(td); });
    const td = document.createElement("td"); td.textContent = main; if (CFG.task !== "multiclass") td.className = "num"; tr.append(td);
    if (extra) {
      const td2 = document.createElement("td");
      if (Array.isArray(extra)) { const tag = document.createElement("span"); tag.className = "tag " + extra[0]; tag.textContent = extra[1]; td2.append(tag); }
      else td2.textContent = extra;
      tr.append(td2);
    }
    if (CFG.task === "binary" && r.p.prediction === CFG.positive_label) flagged++;
    body.append(tr);
  }
  const n = results.length.toLocaleString();
  $("many-summary").textContent = CFG.task === "binary" ? n + " rows scored; " + flagged.toLocaleString() + " predicted " + CFG.positive_label + ". Highest score first."
    : CFG.task === "multiclass" ? n + " rows predicted. Most confident first." : n + " rows predicted. Highest value first.";
  $("many-out").hidden = false;
}

$("download").addEventListener("click", () => {
  if (!results) return;
  let extra;
  if (CFG.task === "regression") extra = ["prediction"];
  else if (CFG.task === "multiclass") extra = ["prediction", "confidence", ...CFG.classes.map(c => "probability_" + c)];
  else extra = ["score", "prediction", "level"];
  const lines = [["id", ...NAMES, ...extra].map(csvEscape).join(",")];
  for (const r of results) {
    let tail;
    if (CFG.task === "regression") tail = [r.p.prediction];
    else if (CFG.task === "multiclass") { const pr = r.p.probabilities || {}; tail = [r.p.prediction, Math.max(...Object.values(pr)), ...CFG.classes.map(c => pr[c])]; }
    else { const prob = r.p.probability ?? r.p.churn_probability; tail = [prob.toFixed(4), r.p.prediction, level(prob, r.p.prediction === CFG.positive_label)[1]]; }
    lines.push([r.id, ...NAMES.map(n => r.c[n]), ...tail].map(csvEscape).join(","));
  }
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/csv"}));
  a.download = "predictions.csv"; a.click(); URL.revokeObjectURL(a.href);
});

fillForm(CFG.presets[0].values);
checkStatus();
</script>
</body>
</html>
"""
