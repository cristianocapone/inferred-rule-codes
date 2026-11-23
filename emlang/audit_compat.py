"""Write a group audit (emlang/audit_group.py) in the format the evaluation scripts expect.

emlang/llm_game.py --audit and emlang/eval_clean.py --audit read {"rows": [{"prog": ...}], "novel_progs": [...]}.
The group audit reports three nested levels of novelty; which one is right depends on what the run saw:
  direct    the run did not augment its episodes (emlang/llm_game.py), so only the training programs themselves
            (single operations, trained pairs, self-compositions and trained triples) had to be excluded
  dihedral  augmentation by the eight dihedral transforms only
  group     the full augmentation of training, dihedral transforms and colour permutations (emlang/train.py)
Usage: python emlang/audit_compat.py --group audit_20260917/group_audit_A_hold2.json --level direct --out ...json
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True); ap.add_argument("--level", choices=["direct", "dihedral", "group"], default="direct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.group))
    novel = d[f"novel_{a.level}"]
    rep = dict(split=d["split"], library=d["library"], level=a.level, source=os.path.basename(a.group),
               n_heldout=d["n_heldout"], n_functionally_novel=len(novel),
               novel_progs=novel, rows=[dict(prog=r["prog"], names=r["names"], novel=r["prog"] in novel) for r in d["rows"]])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"[audit-compat] {a.out}: {len(novel)} of {d['n_heldout']} {d['split']} programs novel at level '{a.level}'")


if __name__ == "__main__":
    main()
