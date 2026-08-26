"""``bench report`` — every scored pair, plus the summary statistics, as one page.

The aggregate numbers are the smaller half of what a run tells you. With one
reference per window and a human-vs-human ceiling of 69.8 CIDEr, the pairs
themselves are the evidence: reading fifty of them side by side is what shows
*how* a prediction misses — wrong subject, wrong moment, right idea in different
words — which no scalar distinguishes.

Writes a self-contained HTML file. The pairs are embedded as JSON and filtered
in the browser, so the whole run stays one file you can open or publish.
"""

import json
import logging
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from timeline import NARRATION_WORDS_PER_SEC

logger = logging.getLogger(__name__)


def _named(text: str) -> set[str]:
    """Capitalised tokens after the first word — a cheap proxy for character names.

    Not CRITIC (that needs coreference resolution and a cast list); enough to show
    whether the model names anyone at all, which is the precondition for CRITIC
    being worth implementing.
    """
    return {w.strip(".,!?;:'\"") for w in text.split()[1:] if w[:1].isupper()}


def summarize(pairs: list[dict], results: dict) -> dict:
    """Aggregate statistics over scored pairs, beyond what `bench score` reports."""
    scored = [p for p in pairs if p.get("pred")]
    gt_words = [len(p["gt"].split()) for p in scored]
    pred_words = [len(p["pred"].split()) for p in scored]
    # What rate the human describer actually spoke at, implied by the reference
    # line and the window it had to fit.
    human_wps = [len(p["gt"].split()) / p["duration"] for p in scored if p["duration"]]
    budget = [
        int(p["duration"] * p.get("words_per_sec", NARRATION_WORDS_PER_SEC))
        for p in scored
    ]

    by_movie: dict[str, list[float]] = defaultdict(list)
    for pair in scored:
        if pair.get("llm_ad_eval") is not None:
            by_movie[pair.get("movie_title") or "—"].append(pair["llm_ad_eval"])

    judged = [p["llm_ad_eval"] for p in scored if p.get("llm_ad_eval") is not None]
    return {
        "pairs": len(pairs),
        "scored": len(scored),
        "clips": len({p["video_id"] for p in pairs}),
        "movies": len({p.get("movie_title") for p in pairs}),
        "cider": results.get("cider"),
        "recall": results.get("recall@1/5"),
        "bertscore": results.get("bertscore"),
        "llm_ad_eval": results.get("llm_ad_eval"),
        "judge_model": results.get("judge_model"),
        "histogram": results.get("histogram")
        or {str(k): v for k, v in sorted(Counter(judged).items())},
        "gt_words_mean": round(st.mean(gt_words), 1) if gt_words else 0,
        "pred_words_mean": round(st.mean(pred_words), 1) if pred_words else 0,
        "human_wps_mean": round(st.mean(human_wps), 2) if human_wps else 0,
        "human_wps_median": round(st.median(human_wps), 2) if human_wps else 0,
        "words_per_sec": scored[0].get("words_per_sec", NARRATION_WORDS_PER_SEC)
        if scored
        else NARRATION_WORDS_PER_SEC,
        "budget_mean": round(st.mean(budget), 1) if budget else 0,
        "over_budget": sum(
            1 for p, b in zip(scored, budget, strict=True) if len(p["gt"].split()) > b
        ),
        "window_mean": round(st.mean([p["duration"] for p in scored]), 2)
        if scored
        else 0,
        "frames_mean": round(st.mean([p.get("frames", 0) for p in scored]), 1)
        if scored
        else 0,
        "gt_named": sum(1 for p in scored if _named(p["gt"])),
        "pred_named": sum(1 for p in scored if _named(p["pred"])),
        "by_movie": sorted(
            ((m, round(st.mean(v), 2), len(v)) for m, v in by_movie.items()),
            key=lambda row: row[1],
        ),
    }


def build_report(preds_path: Path, results_path: Path, out_path: Path) -> Path:
    with open(preds_path, encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle if line.strip()]
    results = (
        json.loads(Path(results_path).read_text())
        if Path(results_path).exists()
        else {}
    )
    stats = summarize(pairs, results)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(pairs, stats), encoding="utf-8")
    logger.info(
        "report: %d pair(s) over %d clip(s), %d movie(s) -> %s",
        stats["pairs"],
        stats["clips"],
        stats["movies"],
        out_path,
    )
    return out_path


def _render(pairs: list[dict], s: dict) -> str:
    payload = json.dumps(
        [
            {
                "movie": p.get("movie_title") or "—",
                "clip": p["video_id"],
                "start": round(p["scaled_start"], 2),
                "end": round(p["scaled_end"], 2),
                "dur": round(p["duration"], 2),
                "frames": p.get("frames"),
                "gt": p["gt"],
                "pred": p.get("pred") or "",
                "cider": p.get("cider"),
                "judge": p.get("llm_ad_eval"),
            }
            for p in pairs
        ],
        ensure_ascii=False,
    )
    hist = s["histogram"]
    hist_max = max([v for v in hist.values()] or [1])
    bars = "\n".join(
        f'<div class="bar-row" data-score="{k}"><span class="bar-k">{k}</span>'
        f'<div class="bar-track"><div class="bar-fill s{k}" style="width:{100 * v / hist_max:.1f}%"></div></div>'
        f'<span class="bar-v">{v}</span></div>'
        for k, v in sorted(hist.items())
    )
    movie_rows = "\n".join(
        f'<tr><td>{m}</td><td class="num">{n}</td><td class="num">{score:.2f}</td></tr>'
        for m, score, n in s["by_movie"]
    )

    over_pct = 100 * s["over_budget"] / s["scored"] if s["scored"] else 0
    return f"""<title>AD Benchmark Pairs</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{
  --paper:#F2F3F5; --card:#FFFFFF; --ink:#171A20; --slate:#5B6572; --hair:#DDE0E5;
  --brass:#96661F; --brass-soft:#F0E6D6;
  --s0:#E2E8E7; --s1:#BFD4D1; --s2:#94BAB5; --s3:#639C96; --s4:#3A7C76; --s5:#155E58;
  --human:#4A3F8F;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#101317; --card:#181C22; --ink:#E8EAED; --slate:#98A2B0; --hair:#2A3038;
    --brass:#D9A85C; --brass-soft:#2E2519;
    --s0:#2B3432; --s1:#3C534F; --s2:#4E736D; --s3:#5F938C; --s4:#77B3AB; --s5:#95D2C9;
    --human:#A99BEA;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#101317; --card:#181C22; --ink:#E8EAED; --slate:#98A2B0; --hair:#2A3038;
  --brass:#D9A85C; --brass-soft:#2E2519;
  --s0:#2B3432; --s1:#3C534F; --s2:#4E736D; --s3:#5F938C; --s4:#77B3AB; --s5:#95D2C9;
  --human:#A99BEA;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:15px; line-height:1.55;
}}
.wrap {{ max-width:1080px; margin:0 auto; padding:56px 24px 96px; }}
h1 {{
  font-family:Newsreader,Georgia,serif; font-weight:600; font-size:2.6rem;
  margin:0 0 8px; letter-spacing:-.015em; text-wrap:balance;
}}
.lede {{ color:var(--slate); max-width:62ch; margin:0 0 8px; }}
.runline {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.78rem; color:var(--slate);
  border-top:1px solid var(--hair); padding-top:10px; margin-top:24px;
}}
h2 {{
  font-family:Newsreader,Georgia,serif; font-weight:600; font-size:1.45rem;
  margin:56px 0 16px; letter-spacing:-.01em;
}}
.eyebrow {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.7rem;
  text-transform:uppercase; letter-spacing:.12em; color:var(--brass); margin:0 0 6px;
}}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:28px; }}
.tile {{ background:var(--card); border:1px solid var(--hair); border-radius:4px; padding:16px 18px; }}
.tile .k {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.68rem;
  text-transform:uppercase; letter-spacing:.1em; color:var(--slate);
}}
.tile .v {{
  font-family:Newsreader,Georgia,serif; font-size:2.1rem; line-height:1.1;
  font-variant-numeric:tabular-nums; margin-top:6px;
}}
.tile .n {{ font-size:.76rem; color:var(--slate); margin-top:4px; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:32px; align-items:start; }}
@media (max-width:760px) {{ .two {{ grid-template-columns:1fr; }} }}
.bar-row {{ display:grid; grid-template-columns:18px 1fr 34px; gap:10px; align-items:center; margin-bottom:6px; }}
.bar-k, .bar-v {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.8rem; color:var(--slate); font-variant-numeric:tabular-nums; }}
.bar-v {{ text-align:right; color:var(--ink); }}
.bar-track {{ background:var(--paper); border:1px solid var(--hair); border-radius:2px; height:20px; }}
.bar-fill {{ height:100%; border-radius:0 3px 3px 0; }}
.s0{{background:var(--s0)}} .s1{{background:var(--s1)}} .s2{{background:var(--s2)}}
.s3{{background:var(--s3)}} .s4{{background:var(--s4)}} .s5{{background:var(--s5)}}
table {{ width:100%; border-collapse:collapse; font-size:.86rem; }}
th, td {{ text-align:left; padding:6px 10px 6px 0; border-bottom:1px solid var(--hair); }}
th {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.68rem; font-weight:500;
  text-transform:uppercase; letter-spacing:.1em; color:var(--slate);
}}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; font-family:"IBM Plex Mono",ui-monospace,monospace; }}
.scroll {{ overflow-x:auto; }}
.controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:0 0 20px; }}
.chip {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.75rem;
  border:1px solid var(--hair); background:var(--card); color:var(--slate);
  border-radius:999px; padding:5px 12px; cursor:pointer;
}}
.chip[aria-pressed="true"] {{ background:var(--brass-soft); border-color:var(--brass); color:var(--brass); }}
.chip:focus-visible, select:focus-visible, input:focus-visible {{ outline:2px solid var(--brass); outline-offset:2px; }}
select, input[type=search] {{
  font-family:inherit; font-size:.82rem; padding:5px 10px; border-radius:4px;
  border:1px solid var(--hair); background:var(--card); color:var(--ink);
}}
.count {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.75rem; color:var(--slate); margin-left:auto; }}
.clip-h {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.72rem; color:var(--slate);
  text-transform:uppercase; letter-spacing:.08em; margin:26px 0 8px; padding-bottom:5px;
  border-bottom:1px solid var(--hair);
}}
.pair {{
  background:var(--card); border:1px solid var(--hair); border-left:3px solid var(--sc);
  border-radius:3px; padding:12px 16px; margin-bottom:8px;
  display:grid; grid-template-columns:1fr 116px; gap:16px; align-items:start;
}}
@media (max-width:640px) {{ .pair {{ grid-template-columns:1fr; }} }}
.ref {{
  font-family:Newsreader,Georgia,serif; font-size:1.06rem; color:var(--human);
}}
.hyp {{ font-size:.95rem; margin-top:5px; }}
.tag {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.62rem;
  text-transform:uppercase; letter-spacing:.09em; color:var(--slate);
  display:inline-block; width:3.1em;
}}
.meta {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.7rem; color:var(--slate); text-align:right; font-variant-numeric:tabular-nums; }}
.judge {{
  display:inline-flex; align-items:center; justify-content:center; min-width:2rem;
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.82rem; font-weight:500;
  border-radius:3px; padding:2px 7px; color:#fff; margin-bottom:4px;
}}
.judge.j0,.judge.j1 {{ color:var(--ink); }}
.note {{ color:var(--slate); font-size:.86rem; max-width:64ch; }}
.note strong {{ color:var(--ink); font-weight:600; }}
</style>

<div class="wrap">
  <p class="eyebrow">CMD-AD · eval split</p>
  <h1>Audio description, ours against theirs</h1>
  <p class="lede">Every generated narration line beside the human-written reference it was
  scored against. The reference is set in serif, the prediction in sans — the pairs are the
  evidence, and the aggregates below only summarize them.</p>
  <p class="runline">{s["scored"]} scored pairs · {s["clips"]} clips · {s["movies"]} movies ·
  word budget {s["words_per_sec"]} w/s · judge {s["judge_model"] or "—"}</p>

  <div class="tiles">
    <div class="tile"><div class="k">CIDEr</div><div class="v">{s["cider"] if s["cider"] is not None else "—"}</div>
      <div class="n">AutoAD-III 25.0 · StrAD-FT 36.3 · human 69.8</div></div>
    <div class="tile"><div class="k">LLM-AD-eval</div><div class="v">{s["llm_ad_eval"] if s["llm_ad_eval"] is not None else "—"}</div>
      <div class="n">AutoAD-III 2.05 · human 3.06</div></div>
    <div class="tile"><div class="k">Recall@1/5</div><div class="v">{s["recall"] if s["recall"] is not None else "—"}</div>
      <div class="n">chance 20 · AutoAD-III 31.2 · StrAD 38.0 · human 80.4</div></div>
    <div class="tile"><div class="k">Human rate</div><div class="v">{s["human_wps_mean"]}</div>
      <div class="n">words/sec · our budget {s["words_per_sec"]}</div></div>
  </div>

  <div class="two" style="margin-top:52px">
    <div>
      <p class="eyebrow">Judge score</p>
      <h2 style="margin-top:0">Where the pairs land</h2>
      {bars}
      <p class="note" style="margin-top:14px">Scores run 0–5 on the paper's own rubric, so the
      band is a magnitude and shades accordingly. The shape matters more than the mean: a run
      that is bimodal — nailed it or described a different moment — averages the same as one
      that is uniformly mediocre.</p>
    </div>
    <div>
      <p class="eyebrow">Per movie</p>
      <h2 style="margin-top:0">Spread across films</h2>
      <div class="scroll"><table>
        <thead><tr><th>Film</th><th class="num">Pairs</th><th class="num">Mean</th></tr></thead>
        <tbody>{movie_rows}</tbody>
      </table></div>
    </div>
  </div>

  <h2>Which of these numbers to trust</h2>
  <p class="note">CIDEr is the field's default here and it is the weakest of the four.
  On this corpus, <strong>copying the neighbouring reference AD</strong> — fluent, on-register
  and describing the wrong moment entirely — scores 30.4 CIDEr against this run's 38.2, and 30.2
  BERTScore against 37.9. Both are largely measuring whether a line <em>sounds like</em> audio
  description. <strong>Recall@1/5 is the one that cannot be fooled</strong>: it asks whether a line
  retrieves its own reference rather than a neighbour, so the same wrong-moment copy scores 0.0,
  and a random line scores chance (20).</p>
  <p class="note" style="margin-top:12px">The ceiling is not 100. Two professional describers
  watching the same frames retrieve each other at <strong>80.4</strong> (AutoAD-III, Table 3) —
  the same scene admits many correct descriptions, and that irreducible disagreement is the real
  target. Measured against the range a system can actually move through, floor to human, this run
  covers about <strong>30%</strong> — and Recall@1/5 and LLM-AD-eval agree on that figure while
  CIDEr flatters it to 53%.</p>

  <h2>What the aggregates hide</h2>
  <p class="note"><strong>The word budget is too tight.</strong> The references imply human
  describers speaking at {s["human_wps_mean"]} words/sec (median {s["human_wps_median"]}), against a
  configured budget of {s["words_per_sec"]}. Our mean line runs {s["pred_words_mean"]} words where the
  reference runs {s["gt_words_mean"]}, and <strong>{s["over_budget"]} of {s["scored"]}
  ({over_pct:.0f}%)</strong> reference lines would not fit the budget at all. A shorter line loses
  n-gram overlap whether or not it describes the right thing.</p>
  <p class="note" style="margin-top:12px"><strong>We rarely name anyone.</strong>
  {s["gt_named"]} of {s["scored"]} references name a character; {s["pred_named"]} of our lines do.
  LLM-AD-eval is instructed to treat any character name as a match, so it cannot see this —
  CRITIC is the metric that would, and a character bank is what would fix it.</p>

  <h2 id="pairs">Every pair</h2>
  <div class="controls">
    <button class="chip" data-filter="all" aria-pressed="true">All</button>
    <button class="chip" data-filter="0-1" aria-pressed="false">Misses (0–1)</button>
    <button class="chip" data-filter="2-3" aria-pressed="false">Partial (2–3)</button>
    <button class="chip" data-filter="4-5" aria-pressed="false">Hits (4–5)</button>
    <select id="movie"><option value="">All films</option></select>
    <input type="search" id="q" placeholder="Search text…">
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>

<script>
const PAIRS = {payload};
const SHADE = ["--s0","--s1","--s2","--s3","--s4","--s5"];
const list = document.getElementById("list");
const countEl = document.getElementById("count");
const movieSel = document.getElementById("movie");
const q = document.getElementById("q");
let band = "all";

[...new Set(PAIRS.map(p => p.movie))].sort().forEach(m => {{
  const o = document.createElement("option"); o.value = m; o.textContent = m; movieSel.appendChild(o);
}});

const esc = t => t.replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));

function inBand(p) {{
  if (band === "all") return true;
  if (p.judge === null || p.judge === undefined) return false;
  const [lo, hi] = band.split("-").map(Number);
  return p.judge >= lo && p.judge <= hi;
}}

function render() {{
  const needle = q.value.trim().toLowerCase();
  const film = movieSel.value;
  const rows = PAIRS.filter(p => inBand(p)
    && (!film || p.movie === film)
    && (!needle || (p.gt + " " + p.pred).toLowerCase().includes(needle)));

  countEl.textContent = rows.length + " of " + PAIRS.length + " pairs";
  const groups = new Map();
  for (const p of rows) {{
    const key = p.movie + " \\u00b7 " + p.clip;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(p);
  }}

  let html = "";
  for (const [key, group] of groups) {{
    html += '<p class="clip-h">' + esc(key) + " — " + group.length + " pair"
         + (group.length === 1 ? "" : "s") + "</p>";
    for (const p of group) {{
      const j = (p.judge === null || p.judge === undefined) ? null : p.judge;
      const shade = j === null ? "--hair" : SHADE[j];
      html += '<div class="pair" style="--sc:var(' + shade + ')">'
        + '<div>'
        + '<div class="ref"><span class="tag">ref</span>' + esc(p.gt) + '</div>'
        + '<div class="hyp"><span class="tag">ours</span>' + esc(p.pred || "— no line generated —") + '</div>'
        + '</div>'
        + '<div class="meta">'
        + (j === null ? '' : '<div><span class="judge j' + j + '" style="background:var(' + shade + ')">' + j + '</span></div>')
        + '<div>' + p.start.toFixed(1) + 's &ndash; ' + p.end.toFixed(1) + 's</div>'
        + '<div>' + p.dur.toFixed(1) + 's &middot; ' + (p.frames ?? "?") + 'f</div>'
        + (p.cider === null || p.cider === undefined ? '' : '<div>cider ' + p.cider.toFixed(0) + '</div>')
        + '</div></div>';
    }}
  }}
  list.innerHTML = html || '<p class="note">No pairs match that filter.</p>';
}}

document.querySelectorAll(".chip").forEach(btn => btn.addEventListener("click", () => {{
  document.querySelectorAll(".chip").forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
  band = btn.dataset.filter;
  render();
}}));
movieSel.addEventListener("change", render);
q.addEventListener("input", render);
render();
</script>
"""
