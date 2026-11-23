"""Co-adaptation test (Supplementary Section S5): is a failing code uninformative, or was its executor not coupled to it?

Reads emlang_runs/analysis/coadapt3/<run>_<fresh|blind>.json (emlang/coadapt.py with the group audit). For every
frozen sender: the exact match of the executor trained with it (original), of a new executor trained for 8 000 steps
on its most probable messages (fresh), and of a new executor trained on the same messages permuted across the episodes
of each batch (blind), on the held-out triples that no training program computes even through augmentation (ten
episodes per triple) and on single operations (150 episodes).
Writes paper/supp_tables/coadapt.tex and emlang_runs/analysis/stats/coadapt.json."""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import glob, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SRC = os.path.join(DATA, "analysis", "coadapt3")
LABELS = [(r"cur_rl_sslpercept_nomdl_s\d", "reward, staged"),
          (r"joint_rl_sslpercept_div\d?_s\d", "reward, joint + information term"),
          (r"joint_rl_sslpercept_nomdl_s\d", "reward, joint"),
          (r"joint_rl_sslpercept_chg_s\d", "reward, joint, changed-cell reward"),
          (r"cur_rl_sslpercept_chg_s\d", "reward, staged, changed-cell reward"),
          (r"joint_rl_sslpercept_slow_s\d", "reward, joint, slower exploration"),
          (r"stab_joint_chg_stab_s\d", "reward, joint, changed-cell + stability 1.0 (soft policy)"),
          (r"stab_joint_stab_s\d", "reward, joint + stability 1.0 (soft policy)"),
          (r"cur_gumbel_sslpercept_nomdl_s\d", "gradient, staged"),
          (r"cur_cont_s\d", "continuous, staged")]


def main():
    runs = {}
    for f in sorted(glob.glob(os.path.join(SRC, "*.json"))):
        m = re.fullmatch(r"(.+)_(fresh|blind)", os.path.basename(f)[:-5])
        if not m: continue
        d = json.load(open(f)); last = d["curve"][-1]
        if last["step"] != d["steps"]: continue                      # unfinished
        r = runs.setdefault(m.group(1), {}); r["original"] = d["original"]; r[m.group(2)] = last
    out, lines = [], [r"\begin{tabular}{lcccccc}\toprule",
                      r" & \multicolumn{3}{c}{new triples (\%)} & \multicolumn{3}{c}{single operations (\%)} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
                      r"sender (library A) & original & fresh & blind & original & fresh & blind \\ \midrule"]
    for pat, lab in LABELS:
        rs = sorted(k for k in runs if re.fullmatch(pat, k))
        if not rs: continue
        cell = lambda mode, key: " / ".join(f"{100 * runs[k][mode][key]:.1f}" if mode in runs[k] else "--" for k in rs)
        lines.append(f"{lab} & {cell('original', 'triples_audited')} & {cell('fresh', 'triples_audited')} & {cell('blind', 'triples_audited')} & "
                     f"{cell('original', 'singles')} & {cell('fresh', 'singles')} & {cell('blind', 'singles')} \\\\")
        out.append(dict(condition=lab, runs=rs, **{mode: [runs[k].get(mode) for k in rs] for mode in ("original", "fresh", "blind")}))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABLES, "coadapt.tex"), "w").write("\n".join(lines) + "\n")
    os.makedirs(os.path.join(RESULTS), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "coadapt.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
