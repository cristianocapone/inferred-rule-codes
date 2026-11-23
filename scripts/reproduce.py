#!/usr/bin/env python3
"""Reproduce the paper from the frozen data exports; no GPU or network required."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def verify_inputs():
    manifest = ROOT / "data/manifest.json"
    if not manifest.exists():
        raise RuntimeError("Missing data/manifest.json; use a complete repository checkout.")
    entries = json.loads(manifest.read_text())["files"]
    failures = []
    for item in entries:
        path = ROOT / item["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            failures.append(item["path"])
    if failures:
        raise RuntimeError("Missing or changed frozen inputs:\n" + "\n".join(failures))
    print(f"Verified {len(entries)} frozen input files.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["all", "figures", "tables", "pdf", "verify"], nargs="?", default="all")
    parser.add_argument("--record-inputs", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.record_inputs:
        verify_inputs()
    if args.target == "verify":
        return
    build = ROOT / "build"
    logs = build / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for directory in [ROOT / "results", ROOT / "paper/figures", ROOT / "paper/supp_tables", ROOT / "pdf"]:
        directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(EMLANG_CANVAS="12", MPLBACKEND="Agg", MPLCONFIGDIR=str(build / "matplotlib"), PYTHONHASHSEED="0")
    if args.record_inputs:
        env["PAPER_TRACE_INPUTS"] = str(build / "input-traces")
    outcomes = []

    def run(label, command, cwd=ROOT, extra_env=None):
        print(f"[{label}]", flush=True)
        started = time.monotonic()
        with (logs / (label + ".log")).open("w") as stream:
            result = subprocess.run(command, cwd=cwd, env={**env, **(extra_env or {})}, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print((logs / (label + ".log")).read_text()[-7000:], file=sys.stderr)
            raise RuntimeError(f"{label} failed. See build/logs/{label}.log")
        outcomes.append({"step": label, "seconds": round(time.monotonic() - started, 2)})

    def python(name):
        run(name, [sys.executable, str(ROOT / "analysis" / (name + ".py"))])

    if args.target in ["all", "tables"]:
        for name in ["info_competence", "coadapt_table", "llm_generality", "stats", "make_supp_tables"]:
            python(name)
    if args.target in ["all", "figures"]:
        if args.target == "figures":
            python("info_competence")
        for name in ["make_fig1", "make_fig2", "make_fig3", "make_fig4", "make_fig_kirby", "make_cover_fig"]:
            python(name)
    if args.target in ["all", "pdf"]:
        for tool in ["pdflatex", "bibtex"]:
            if shutil.which(tool) is None:
                raise RuntimeError(f"{tool} is missing. Install TeX Live or MacTeX; see README.md.")
        python("make_nc")
        tex = build / "tex"
        tex.mkdir(parents=True, exist_ok=True)
        paper = ROOT / "paper"
        bibenv = {"BIBINPUTS": str(paper) + os.pathsep, "BSTINPUTS": str(paper) + os.pathsep}
        for name, output in [("main_nc", "manuscript"), ("main_nature", "manuscript-nature"), ("supplement", "supplement")]:
            command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-no-shell-escape", f"-output-directory={tex}", name + ".tex"]
            run(name + "-latex1", command, paper)
            run(name + "-bibtex", ["bibtex", name], tex, bibenv)
            run(name + "-latex2", command, paper)
            run(name + "-latex3", command, paper)
            log = (tex / (name + ".log")).read_text(errors="replace")
            if "There were undefined references" in log or "There were undefined citations" in log:
                raise RuntimeError(f"Unresolved references in {name}; see build/tex/{name}.log")
            shutil.copy2(tex / (name + ".pdf"), ROOT / "pdf" / (output + ".pdf"))
    (build / "last-run.json").write_text(json.dumps({"python": sys.version, "target": args.target, "steps": outcomes}, indent=2) + "\n")
    print(f"Completed {args.target}. Outputs: paper/figures/, paper/supp_tables/, results/, pdf/", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        sys.exit(str(error))
