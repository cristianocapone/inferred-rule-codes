"""Human side of the LARC comparison (Acquaviva et al., NeurIPS 2022 D&B).

LARC is our speaker-executor game played by people with English as the channel: a *describer*
solves an ARC-1 task and writes the rule; *builders* must then solve a NEW input from the
description alone. Per ARC-1 task we extract how transmissible the rule is for humans.

Outputs a CSV/JSON keyed by ARC-1 task id:
  build_success        fraction of builder attempts that produced the correct grid
  n_builds             number of builder attempts
  desc_len_words       mean length of the full description (see + do + grid)
  desc_len_do          mean length of the "to make the output you have to..." part (the rule itself)
  confidence           mean describer confidence (0-10)
  verif_success        fraction of describers who could solve the test input themselves
  n_desc               number of descriptions
"""
import argparse, glob, json, os, statistics as st

LARC_DIR = os.environ.get("LARC_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_rules", "LARC"))


def words(s):
    return len((s or "").split())


def task_stats(path):
    d = json.load(open(path))
    tid = d["name"].replace(".json", "")
    descs = list(d.get("descriptions", {}).values())
    if not descs:
        return None
    succ, att, dl, dd, conf, ver = [], 0, [], [], [], []
    for de in descs:
        builds = list((de.get("builds") or {}).values())
        for b in builds:
            succ.append(1.0 if b.get("success") else 0.0)
            att += int(b.get("num_attempts") or 0)
        dl.append(words(de.get("see_description")) + words(de.get("do_description")) + words(de.get("grid_description")))
        dd.append(words(de.get("do_description")))
        if de.get("confidence") is not None:
            try: conf.append(float(de["confidence"]))
            except (TypeError, ValueError): pass
        ver.append(1.0 if de.get("succeeded_verification") else 0.0)
    return dict(task=tid, n_desc=len(descs), n_builds=len(succ),
                build_success=(sum(succ) / len(succ)) if succ else None,
                desc_len_words=st.mean(dl) if dl else None,
                desc_len_do=st.mean(dd) if dd else None,
                confidence=st.mean(conf) if conf else None,
                verif_success=st.mean(ver) if ver else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="emlang_runs/larc_task_stats.json")
    a = ap.parse_args()
    rows = [r for r in (task_stats(f) for f in sorted(glob.glob(os.path.join(LARC_DIR, "dataset", "tasks_json", "*.json")))) if r]
    json.dump({r["task"]: r for r in rows}, open(a.out, "w"), indent=1)
    bs = [r["build_success"] for r in rows if r["build_success"] is not None]
    dl = [r["desc_len_do"] for r in rows if r["desc_len_do"] is not None]
    print(f"{len(rows)} ARC-1 tasks | builds per task {st.mean([r['n_builds'] for r in rows]):.1f} | "
          f"build success mean {st.mean(bs):.3f} (min {min(bs):.2f} max {max(bs):.2f}) | "
          f"rule-description length {st.mean(dl):.1f} words (min {min(dl):.0f} max {max(dl):.0f})")
    print("saved", a.out)


if __name__ == "__main__":
    main()
