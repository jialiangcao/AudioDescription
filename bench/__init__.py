"""Offline CMD-AD benchmark harness.

Runs this project's narration pipeline over the Condensed Movies clips from the
CMD-AD dataset (*AutoAD III: The Prequel*, arXiv 2404.14412) and scores the
output against the human-written audio description in the CSV.

Deliberately outside ``src/`` and in neither Docker image: it is a developer
tool, not part of the deployed service, and nothing under ``src/`` may import
it. It goes the other way — it calls the pipeline's stage functions directly,
against a purely local ``JobBlobs``, so a run needs no Postgres, no Redis and no
object storage. The only credential it wants is ``GEMINI_API_KEY``.

    uv sync --extra media --extra bench
    uv run python -m bench fetch --csv data/cmd_ad.csv --limit 3
    uv run python -m bench run   --csv data/cmd_ad.csv
    uv run python -m bench score --preds data/preds.jsonl
"""

import sys
from pathlib import Path

# ``src/`` is not a package; modules there import each other with flat, absolute
# names, which works because every entry point puts it on sys.path (uvicorn's
# --app-dir, celery's --workdir, pytest's pythonpath). This is that entry
# point's version of the same thing.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
