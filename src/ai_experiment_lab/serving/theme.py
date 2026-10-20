"""Look and feel shared by the two web pages (``/app`` and ``/lab``).

One idea carries the design: a lab bench. Cool grey-green surfaces, deep teal ink, and the three
"litmus" colours (green, amber, magenta) reserved for meaning (low, medium and high risk) so a
colour on the page always says something. Headings use Bricolage Grotesque, text uses IBM Plex
Sans; both fall back to system fonts if the font host is unreachable.
"""

from __future__ import annotations

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&"
    'family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
)

CSS = r"""
:root {
  --bench:#edf1ee; --glass:#fbfdfc; --sunk:#f1f5f3; --ink:#0c2a2b; --muted:#4d6567; --line:#cbd8d4;
  --reagent:#0b6e6b; --reagent-ink:#ffffff; --deep:#072f30; --deep-ink:#e8f5f2; --deep-muted:#9cc3bd;
  --low:#14804f; --mid:#b66a00; --high:#c81e5b;
  --lowbg:#e3f3eb; --midbg:#fcefd9; --highbg:#fce6ee; --focus:#0b6e6b;
  --display:"Bricolage Grotesque","Segoe UI Variable Display","Segoe UI",system-ui,sans-serif;
  --text:"IBM Plex Sans","Segoe UI",system-ui,-apple-system,sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root { --bench:#091415; --glass:#102120; --sunk:#0c1a1a; --ink:#e3efec; --muted:#93b0ac; --line:#24403d;
    --reagent:#46c9bc; --reagent-ink:#04201e; --deep:#061c1d; --deep-ink:#e8f5f2; --deep-muted:#82aea8;
    --low:#4fd391; --mid:#f2b45a; --high:#ff7aa6;
    --lowbg:#0f2a20; --midbg:#2c2210; --highbg:#331622; --focus:#46c9bc; }
}
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; background:var(--bench); color:var(--ink); font:16px/1.55 var(--text); font-variant-numeric:tabular-nums; }
a { color:var(--reagent); text-underline-offset:3px; }
h1, h2, h3 { font-family:var(--display); letter-spacing:-.015em; line-height:1.12; margin:0; }
[hidden] { display:none !important; }
:focus-visible { outline:3px solid var(--focus); outline-offset:2px; border-radius:6px; }

/* top bar */
.top { position:sticky; top:0; z-index:5; background:var(--deep); color:var(--deep-ink); }
.top .in { max-width:1040px; margin:0 auto; padding:0 20px; height:56px; display:flex; align-items:center; gap:24px; }
.brand { display:flex; align-items:center; gap:10px; font:800 1.05rem var(--display); color:inherit; text-decoration:none; letter-spacing:-.01em; }
.brand i { display:block; width:30px; height:10px; border-radius:5px; background:linear-gradient(90deg,#2fbf84,#f0a21c 55%,#ee3d7c); }
.top nav { display:flex; gap:4px; margin-left:auto; }
.top nav a { color:var(--deep-muted); text-decoration:none; padding:7px 14px; border-radius:999px; font-weight:500; font-size:.95rem; }
.top nav a:hover { color:var(--deep-ink); }
.top nav a[aria-current="page"] { background:rgba(255,255,255,.12); color:var(--deep-ink); }

/* hero */
.hero { background:var(--deep); color:var(--deep-ink); padding:44px 0 96px; }
.hero .in { max-width:1040px; margin:0 auto; padding:0 20px; }
.hero h1 { font-size:clamp(2rem,5.2vw,3.4rem); font-weight:800; max-width:16ch; }
.hero p { color:var(--deep-muted); max-width:56ch; margin:14px 0 0; font-size:1.05rem; }
.hero .hint2 { font-size:.88rem; max-width:70ch; margin-top:14px; }
.status { display:inline-flex; align-items:center; gap:8px; margin-top:20px; font-size:.85rem; color:var(--deep-muted); }
.status::before { content:""; width:9px; height:9px; border-radius:50%; background:var(--deep-muted); }
.status.ok { color:#8fe3bd; } .status.ok::before { background:#2fbf84; box-shadow:0 0 0 4px rgba(47,191,132,.2); }
.status.bad { color:#ffb3c9; } .status.bad::before { background:#ee3d7c; }

/* page body */
.wrap { max-width:1040px; margin:-64px auto 0; padding:0 20px 40px; }
.sheet { background:var(--glass); border:1px solid var(--line); border-radius:18px; padding:clamp(16px,3vw,32px); margin-bottom:20px; }
.notice { background:var(--midbg); border-left:4px solid var(--mid); border-radius:6px 12px 12px 6px; padding:12px 16px; margin:0 0 20px; font-size:.92rem; }
.small { color:var(--muted); font-size:.88rem; }
.error { color:var(--high); margin-top:12px; white-space:pre-line; font-weight:500; }
.note { background:var(--midbg); border-left:4px solid var(--mid); border-radius:6px 12px 12px 6px; padding:10px 14px; margin:10px 0; font-size:.94rem; }

/* controls */
label { display:flex; flex-direction:column; gap:5px; font-size:.9rem; }
label > span { font-weight:600; }
.hint { color:var(--muted); font-size:.8rem; font-weight:400; }
select, input[type=number], input[type=text], input[type=file] {
  font:inherit; width:100%; padding:10px 12px; color:var(--ink); background:var(--sunk);
  border:1.5px solid var(--line); border-radius:10px; transition:border-color .15s, background .15s; }
select:hover, input:hover { border-color:var(--muted); }
select:focus, input:focus { border-color:var(--reagent); background:var(--glass); outline:none; box-shadow:0 0 0 3px color-mix(in srgb, var(--reagent) 25%, transparent); }
button { font:inherit; cursor:pointer; }
button.primary, .btn { display:inline-block; font-weight:600; padding:12px 22px; border:0; border-radius:999px; background:var(--reagent); color:var(--reagent-ink); transition:transform .12s, filter .12s; text-decoration:none; }
button.primary:hover { filter:brightness(1.08); transform:translateY(-1px); }
button.secondary { font-weight:500; padding:9px 16px; border:1.5px solid var(--line); border-radius:999px; background:transparent; color:var(--ink); transition:border-color .12s, background .12s; }
button.secondary:hover { border-color:var(--reagent); background:color-mix(in srgb, var(--reagent) 10%, transparent); }
button:disabled { opacity:.5; cursor:progress; transform:none; filter:none; }
.row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }

/* tables */
table { border-collapse:collapse; width:100%; font-size:.92rem; }
th, td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-weight:600; color:var(--muted); font-size:.82rem; background:var(--glass); position:sticky; top:0; }
td.num, td.n { text-align:right; font-variant-numeric:tabular-nums; }
.tag { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem; font-weight:600; }
.tag.low { color:var(--low); background:var(--lowbg); } .tag.mid { color:var(--mid); background:var(--midbg); } .tag.high { color:var(--high); background:var(--highbg); }
.scroll { overflow:auto; border:1px solid var(--line); border-radius:12px; }
.scroll table { margin:0; } .scroll tbody tr:last-child td { border-bottom:0; }

/* the litmus scale: where a score sits between low and high */
.scale { position:relative; height:14px; border-radius:7px; margin:34px 0 8px;
  background:linear-gradient(90deg,#2fbf84 0%,#f0a21c 55%,#ee3d7c 100%); }
.scale .tick { position:absolute; top:-5px; bottom:-5px; width:2px; background:var(--ink); opacity:.55; }
.scale .tick b { position:absolute; top:-20px; left:50%; transform:translateX(-50%); font:500 .72rem var(--text); white-space:nowrap; color:var(--muted); }
.scale .pin { position:absolute; top:-9px; width:6px; height:32px; margin-left:-3px; border-radius:3px; background:var(--ink); box-shadow:0 0 0 3px var(--glass); transition:left .9s cubic-bezier(.2,.8,.2,1); }
.scale-ends { display:flex; justify-content:space-between; font-size:.78rem; color:var(--muted); }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } html { scroll-behavior:auto; } }
@media (max-width:640px) { .top nav a { padding:7px 10px; } .hero { padding:32px 0 80px; } .brand span { display:none; } }
"""


def topbar(active: str) -> str:
    """The bar at the top of every page. ``active`` is ``"predict"`` or ``"advisor"``."""
    def link(key: str, href: str, label: str) -> str:
        cur = ' aria-current="page"' if key == active else ""
        return f'<a href="{href}"{cur}>{label}</a>'

    return ('<header class="top"><div class="in">'
            '<a class="brand" href="lab"><i aria-hidden="true"></i><span>AI Experiment Lab</span></a>'
            f'<nav aria-label="Pages">{link("advisor", "lab", "Check your CSV")}'
            f'{link("predict", "app", "Churn demo")}</nav></div></header>')
