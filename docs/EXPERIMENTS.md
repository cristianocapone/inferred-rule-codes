# Experiment implementation

The `emlang/` package contains the implementation underlying the recorded experiments.
It is independent of the lightweight figure pipeline. Install `requirements-training.txt`
only if you intend to run training or checkpoint-based evaluation; it adds PyTorch and the
pretrained-model dependencies.

| Component | Entry points |
|---|---|
| Synthetic task libraries and splits | `emlang/gen.py`, `emlang/gen_b.py` |
| Sender and executor (the code and the checkpoints call the sender a speaker) | `emlang/model.py` |
| Main communication game | `python -m emlang.train --help` |
| Contrastive perception pretraining | `python -m emlang.pretrain_perception --help` |
| Split and augmentation audit | `python -m emlang.audit_group --help` |
| Checkpoint evaluation and interventions | `python -m emlang.eval_clean --help`, `python -m emlang.probe --help` |
| Fresh-executor transmission | `python -m emlang.transmit --help`, `python -m emlang.coadapt --help` |
| Message decoding | `emlang/decode_naming.py`, `emlang/code_info.py`, `emlang/code_metrics.py` |
| RE-ARC evaluation | `emlang/rearc_data.py`, `emlang/eval_rearc.py` |
| Pretrained-model pilot | `python -m emlang.llm_game --help` |

Always set `EMLANG_CANVAS=12` **before importing the package** for the manuscript's main
grid experiments. The generator's canvas size affects the seeded split, not just padding.
Training flags, stage budgets and architecture settings should follow the manuscript Methods
for the intended condition; the CLI defaults are not a complete recipe for every figure.

Synthetic grids are generated locally. The RE-ARC/ARC resources and pretrained Qwen model
must be obtained separately from their original providers. No cluster launchers or credentials
are included. Save new checkpoints and logs outside the immutable `data/` tree, e.g. in
`checkpoints/` and `build/new-runs/`.

Optional training dependencies are not an exact lock of the historical GPU environment.
The tested, pinned environment in `requirements.txt` applies to figure/statistics reproduction.
