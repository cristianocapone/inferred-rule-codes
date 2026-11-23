"""Repository-local paths and optional input tracing for the reproduction bundle."""
from pathlib import Path
import atexit
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "supp_tables"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

if os.environ.get("PAPER_TRACE_INPUTS"):
    _reads = set()

    def _audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1]
        if mode is not None and ("w" in mode or "a" in mode or "+" in mode):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(DATA):
            _reads.add(str(path.relative_to(ROOT)))

    sys.addaudithook(_audit)

    def _save_reads():
        target = Path(os.environ["PAPER_TRACE_INPUTS"])
        target.mkdir(parents=True, exist_ok=True)
        (target / (Path(sys.argv[0]).stem + ".json")).write_text(
            json.dumps(sorted(_reads), indent=2) + "\n"
        )

    atexit.register(_save_reads)
