"""``bench qa-report`` — accuracy, its controls, and every answered question.

Scoring multiple choice needs no model and no API call: it is a comparison. What
does need care is the *reading*, so the report puts three things next to the
headline number — chance (25% on 4-way), the blind control, and a bootstrap
confidence interval — because at benchmark sizes we can afford, the interval is
usually wide enough to swallow the difference anyone would want to claim.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Video-MME is 4-way; CinePile is 5-way. Chance is the floor a result is read
# against, so it is derived per run rather than assumed.
DEFAULT_CHOICES = 4


def is_unavailable(record: dict) -> bool:
    """Did this question never get a measurement because the video was gone?

    Video-MME's YouTube sources rot: a share of them are now removed or private.
    Such a question is a missing measurement, not a wrong answer — counting it
    wrong would penalise the video modes against the blind control, which needs
    no video at all, and make the two numbers incomparable.

    The flag is written by ``qa_run`` now; the fallback recognises records from
    before it existed, which are identifiable by having no response whatsoever.
    """
    if record.get("video_unavailable"):
        return True
    return (
        record.get("mode") != "blind"
        and record.get("cycles") is None
        and not (record.get("raw") or "")
    )


def load_runs(paths: list[Path]) -> dict[str, list[dict]]:
    """Prediction records grouped by mode, so the controls sit beside the run."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        if not Path(path).exists():
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    runs[record.get("mode", "agent")].append(record)
    return dict(runs)


def _bootstrap(flags: list[bool], resamples: int = 4000) -> tuple[float, float]:
    """95% CI for a proportion, by resampling the questions."""
    if not flags:
        return (0.0, 0.0)
    random.seed(0)
    n = len(flags)
    means = sorted(
        100 * sum(random.choice(flags) for _ in range(n)) / n for _ in range(resamples)
    )
    return round(means[int(0.025 * resamples)], 1), round(
        means[int(0.975 * resamples) - 1], 1
    )


def summarize_run(records: list[dict], only: set[str] | None = None) -> dict:
    """Accuracy over the questions that were actually measurable.

    ``only`` restricts to a set of question ids — used to score every mode over
    the *same* questions, without which a mode that lost videos to dead links is
    being compared on a harder set than one that did not.
    """
    unavailable = [r for r in records if is_unavailable(r)]
    records = [r for r in records if not is_unavailable(r)]
    if only is not None:
        records = [r for r in records if r["question_id"] in only]
    answered = [r for r in records if r.get("predicted") is not None]
    flags = [bool(r["correct"]) for r in records]
    lo, hi = _bootstrap(flags)
    choices = max(
        (len(r.get("options") or []) for r in records), default=DEFAULT_CHOICES
    )

    by_task: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        by_task[record.get("task_type") or "—"].append(bool(record["correct"]))

    return {
        "questions": len(records),
        "unavailable": len(unavailable),
        # A question the agent never committed to a letter on is scored wrong,
        # not dropped: refusing to answer is not the same as being right, and
        # dropping them would quietly inflate the accuracy.
        "unparsed": len(records) - len(answered),
        "videos": len({r["video_id"] for r in records}),
        "accuracy": round(100 * sum(flags) / len(flags), 1) if flags else 0.0,
        "ci": [lo, hi],
        "chance": round(100 / choices, 1),
        "mean_seconds": round(
            sum(r.get("seconds") or 0 for r in records) / len(records), 1
        )
        if records
        else 0.0,
        "mean_cycles": round(
            sum(r.get("cycles") or 0 for r in records) / len(records), 1
        )
        if records
        else 0.0,
        "by_task": sorted(
            (
                (task, round(100 * sum(v) / len(v), 1), len(v))
                for task, v in by_task.items()
            ),
            key=lambda row: -row[1],
        ),
    }


def common_questions(runs: dict[str, list[dict]]) -> set[str]:
    """Question ids every mode actually measured — the only fair comparison set."""
    sets = [
        {r["question_id"] for r in records if not is_unavailable(r)}
        for records in runs.values()
    ]
    return set.intersection(*sets) if sets else set()


def build_report(preds_paths: list[Path], out_path: Path) -> Path:
    runs = load_runs(preds_paths)
    if not runs:
        raise FileNotFoundError("no prediction files with any records")
    shared = common_questions(runs) if len(runs) > 1 else None
    stats = {
        mode: summarize_run(records, only=shared) for mode, records in runs.items()
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(runs, stats), encoding="utf-8")
    logger.info("qa-report: %s -> %s", ", ".join(stats), out_path)
    return out_path


MODE_LABEL = {
    "agent": "Full agent",
    "blind": "Blind (no video)",
    "subtitles": "Transcript only",
}
MODE_NOTE = {
    "agent": "planner loop, CLIP retrieval, frame inspection",
    "blind": "question and options only — the language-prior floor",
    "subtitles": "extracted dialogue, no pixels",
}


def _render(runs: dict[str, list[dict]], stats: dict[str, dict]) -> str:
    order = [m for m in ("agent", "subtitles", "blind") if m in stats]
    primary = stats.get("agent") or stats[order[0]]
    chance = primary["chance"]

    tiles = "\n".join(
        f"""<div class="tile{" lead" if mode == "agent" else ""}">
      <div class="k">{MODE_LABEL.get(mode, mode)}</div>
      <div class="v">{stats[mode]["accuracy"]}<span class="pct">%</span></div>
      <div class="ci">95% CI {stats[mode]["ci"][0]}–{stats[mode]["ci"][1]}</div>
      <div class="n">{MODE_NOTE.get(mode, "")}</div></div>"""
        for mode in order
    )

    task_rows = "\n".join(
        f'<tr><td>{task}</td><td class="num">{n}</td><td class="num">{acc}</td>'
        f'<td class="bar"><span style="width:{acc:.0f}%"></span></td></tr>'
        for task, acc, n in primary["by_task"]
    )

    payload = json.dumps(
        [
            {
                "mode": r.get("mode"),
                "vid": r["video_id"],
                "task": r.get("task_type") or "—",
                "q": r["question"],
                "opts": r.get("options") or [],
                "gold": r["answer"],
                "pred": r.get("predicted"),
                "ok": bool(r["correct"]),
                "secs": r.get("seconds"),
                "cycles": r.get("cycles"),
                "raw": r.get("raw") or "",
            }
            for r in runs.get("agent", next(iter(runs.values())))
        ],
        ensure_ascii=False,
    )

    lift = primary["accuracy"] - (
        stats["blind"]["accuracy"] if "blind" in stats else chance
    )
    return f"""<title>Video Q&amp;A Agent Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{
  --paper:#F2F3F5; --card:#FFFFFF; --ink:#171A20; --slate:#5B6572; --hair:#DDE0E5;
  --brass:#96661F; --brass-soft:#F0E6D6;
  --ok:#1F6F4A; --ok-soft:#DCEDE3; --bad:#8C2F2F; --bad-soft:#F3DEDE;
  --accent:#2E5C8A;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#101317; --card:#181C22; --ink:#E8EAED; --slate:#98A2B0; --hair:#2A3038;
    --brass:#D9A85C; --brass-soft:#2E2519;
    --ok:#6FCB9B; --ok-soft:#152A20; --bad:#E08A8A; --bad-soft:#2C1A1A;
    --accent:#7FAEDC;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#101317; --card:#181C22; --ink:#E8EAED; --slate:#98A2B0; --hair:#2A3038;
  --brass:#D9A85C; --brass-soft:#2E2519;
  --ok:#6FCB9B; --ok-soft:#152A20; --bad:#E08A8A; --bad-soft:#2C1A1A;
  --accent:#7FAEDC;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:15px; line-height:1.55; }}
.wrap {{ max-width:1060px; margin:0 auto; padding:56px 24px 96px; }}
h1 {{ font-family:Newsreader,Georgia,serif; font-weight:600; font-size:2.5rem;
  margin:0 0 8px; letter-spacing:-.015em; text-wrap:balance; }}
h2 {{ font-family:Newsreader,Georgia,serif; font-weight:600; font-size:1.4rem; margin:52px 0 14px; }}
.eyebrow {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.7rem;
  text-transform:uppercase; letter-spacing:.12em; color:var(--brass); margin:0 0 6px; }}
.lede {{ color:var(--slate); max-width:64ch; margin:0; }}
.runline {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.78rem;
  color:var(--slate); border-top:1px solid var(--hair); padding-top:10px; margin-top:24px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin-top:28px; }}
.tile {{ background:var(--card); border:1px solid var(--hair); border-radius:4px; padding:16px 18px; }}
.tile.lead {{ border-color:var(--accent); box-shadow:inset 3px 0 0 var(--accent); }}
.tile .k {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.68rem;
  text-transform:uppercase; letter-spacing:.1em; color:var(--slate); }}
.tile .v {{ font-family:Newsreader,Georgia,serif; font-size:2.3rem; line-height:1.1;
  font-variant-numeric:tabular-nums; margin-top:6px; }}
.tile .pct {{ font-size:1.1rem; color:var(--slate); margin-left:2px; }}
.tile .ci, .tile .n {{ font-size:.75rem; color:var(--slate); margin-top:4px; }}
.tile .ci {{ font-family:"IBM Plex Mono",ui-monospace,monospace; }}
.note {{ color:var(--slate); max-width:66ch; }}
.note strong {{ color:var(--ink); font-weight:600; }}
table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
th, td {{ text-align:left; padding:7px 10px 7px 0; border-bottom:1px solid var(--hair); }}
th {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.68rem; font-weight:500;
  text-transform:uppercase; letter-spacing:.1em; color:var(--slate); }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums;
  font-family:"IBM Plex Mono",ui-monospace,monospace; }}
td.bar {{ width:34%; }}
td.bar span {{ display:block; height:8px; border-radius:2px; background:var(--accent); }}
.scroll {{ overflow-x:auto; }}
.controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:0 0 18px; }}
.chip {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.75rem;
  border:1px solid var(--hair); background:var(--card); color:var(--slate);
  border-radius:999px; padding:5px 12px; cursor:pointer; }}
.chip[aria-pressed="true"] {{ background:var(--brass-soft); border-color:var(--brass); color:var(--brass); }}
.chip:focus-visible, select:focus-visible {{ outline:2px solid var(--brass); outline-offset:2px; }}
select {{ font-family:inherit; font-size:.82rem; padding:5px 10px; border-radius:4px;
  border:1px solid var(--hair); background:var(--card); color:var(--ink); }}
.count {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.75rem;
  color:var(--slate); margin-left:auto; }}
.q {{ background:var(--card); border:1px solid var(--hair); border-left:3px solid var(--sc);
  border-radius:3px; padding:13px 16px; margin-bottom:9px; }}
.q-head {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }}
.verdict {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.68rem;
  text-transform:uppercase; letter-spacing:.08em; padding:2px 8px; border-radius:3px; }}
.verdict.ok {{ background:var(--ok-soft); color:var(--ok); }}
.verdict.bad {{ background:var(--bad-soft); color:var(--bad); }}
.q-meta {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.68rem; color:var(--slate); }}
.q-text {{ font-family:Newsreader,Georgia,serif; font-size:1.08rem; margin:8px 0 10px; }}
.opts {{ display:grid; gap:3px; }}
.opt {{ font-size:.88rem; padding:3px 8px; border-radius:3px; color:var(--slate); }}
.opt.gold {{ background:var(--ok-soft); color:var(--ok); font-weight:500; }}
.opt.picked {{ background:var(--bad-soft); color:var(--bad); }}
.opt.gold.picked {{ background:var(--ok-soft); color:var(--ok); }}
details {{ margin-top:8px; }}
summary {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.7rem;
  color:var(--slate); cursor:pointer; }}
pre {{ white-space:pre-wrap; font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:.78rem; color:var(--slate); background:var(--paper); padding:10px;
  border-radius:3px; overflow-x:auto; margin:8px 0 0; }}
</style>

<div class="wrap">
  <p class="eyebrow">Video-MME · short split</p>
  <h1>Does the Q&amp;A agent actually watch the video?</h1>
  <p class="lede">Multiple-choice accuracy is easy to report and easy to misread — many video
  questions are answerable from language priors alone. The controls below are the finding: what
  the same agent scores with no video, and with dialogue but no pixels.</p>
  <p class="runline">{primary["questions"]} questions · {primary["videos"]} videos ·
  chance {chance}% · mean {primary["mean_seconds"]}s and {primary["mean_cycles"]} planner cycles per question</p>

  <div class="tiles">{tiles}</div>

  <h2>Reading these numbers</h2>
  <p class="note">The agent scores <strong>{primary["accuracy"]}%</strong> against a chance floor of
  {chance}%{"" if "blind" not in stats else f" and a blind control of {stats['blind']['accuracy']}%"}
  — a lift of <strong>{lift:+.1f} points</strong> over
  {"the blind control" if "blind" in stats else "chance"}. That lift, not the headline accuracy, is
  the evidence that the video is being used at all. The confidence intervals come from resampling
  the questions; where two of them overlap, this run cannot separate those conditions.</p>
  <p class="note" style="margin-top:12px">{primary["unparsed"]} answer(s) never committed to a
  letter and are scored <strong>wrong</strong> rather than dropped — refusing to answer is not the
  same as being right, and dropping them would quietly inflate every number above.</p>

  <h2>By question type</h2>
  <div class="scroll"><table>
    <thead><tr><th>Task type</th><th class="num">n</th><th class="num">Acc</th><th></th></tr></thead>
    <tbody>{task_rows}</tbody>
  </table></div>

  <h2>Every question</h2>
  <div class="controls">
    <button class="chip" data-filter="all" aria-pressed="true">All</button>
    <button class="chip" data-filter="wrong" aria-pressed="false">Wrong only</button>
    <button class="chip" data-filter="right" aria-pressed="false">Right only</button>
    <select id="task"><option value="">All task types</option></select>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>

<script>
const ROWS = {payload};
const list = document.getElementById("list");
const countEl = document.getElementById("count");
const taskSel = document.getElementById("task");
let filter = "all";

[...new Set(ROWS.map(r => r.task))].sort().forEach(t => {{
  const o = document.createElement("option"); o.value = t; o.textContent = t; taskSel.appendChild(o);
}});
const esc = t => (t || "").replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));

function render() {{
  const task = taskSel.value;
  const rows = ROWS.filter(r =>
    (filter === "all" || (filter === "wrong") !== r.ok) && (!task || r.task === task));
  countEl.textContent = rows.length + " of " + ROWS.length + " questions";
  list.innerHTML = rows.map(r => {{
    const opts = r.opts.map(o => {{
      const letter = o.trim().charAt(0);
      const cls = (letter === r.gold ? " gold" : "") + (letter === r.pred ? " picked" : "");
      return '<div class="opt' + cls + '">' + esc(o) + '</div>';
    }}).join("");
    return '<div class="q" style="--sc:var(' + (r.ok ? "--ok" : "--bad") + ')">'
      + '<div class="q-head">'
      + '<span class="verdict ' + (r.ok ? "ok" : "bad") + '">' + (r.ok ? "correct" : "wrong") + '</span>'
      + '<span class="q-meta">' + esc(r.task) + ' &middot; ' + esc(r.vid)
      + ' &middot; gold ' + r.gold + ' &middot; picked ' + (r.pred || "—")
      + (r.secs ? ' &middot; ' + r.secs + 's' : '')
      + (r.cycles ? ' &middot; ' + r.cycles + ' cycles' : '') + '</span></div>'
      + '<div class="q-text">' + esc(r.q) + '</div>'
      + '<div class="opts">' + opts + '</div>'
      + (r.raw ? '<details><summary>agent answer</summary><pre>' + esc(r.raw) + '</pre></details>' : '')
      + '</div>';
  }}).join("") || '<p class="note">No questions match that filter.</p>';
}}

document.querySelectorAll(".chip").forEach(b => b.addEventListener("click", () => {{
  document.querySelectorAll(".chip").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  filter = b.dataset.filter; render();
}}));
taskSel.addEventListener("change", render);
render();
</script>
"""
