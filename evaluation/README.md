# Evaluation

This directory contains evaluation-only tooling. The model inference runner is
documented separately under [`../inference/`](../inference/); generated audio,
API credentials, and result files are not included.

- [`aeq_metric3/`](aeq_metric3/) evaluates whether a generated response
  expresses the requested emotion in speech and text.
- Full-Duplex-Bench evaluation uses a pinned upstream checkout. Its evaluator
  code is not redistributed here.

## Installation

Use Python 3.11 in a dedicated environment for AEQ Metric 3:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r evaluation/requirements.txt
```

Emotion2vec is substantially faster on a compatible GPU. Its model weights are
downloaded by the corresponding library and remain subject to their own
licenses and terms.

## Credentials and outputs

Export `OPENAI_API_KEY` for the AEQ text judge. It also supports an optional
untracked `.env` beside the script; copy `.env.example` locally if needed.
Never commit a populated `.env`.

Generated CSV, JSON, WAV, logs, and result directories are ignored only under
the inference and evaluation paths documented in the repository `.gitignore`.

## Full-Duplex-Bench

We used the upstream
[DanielLin94144/Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
v1.0/v1.5 evaluator at commit
`3e799c45a045256f47d5f1c9cda90157e2d2ec9e` without changing its metric logic.
Clone and pin that revision:

```bash
git clone https://github.com/DanielLin94144/Full-Duplex-Bench.git
cd Full-Duplex-Bench
git checkout 3e799c45a045256f47d5f1c9cda90157e2d2ec9e
python -m venv .venv
source .venv/bin/activate
pip install -r v1_v1.5/requirements.txt
cd v1_v1.5/evaluation
```

Each result sample must contain the generated `output.wav`, an ASR-aligned
`output.json`, and the task-specific annotation files supplied by the
benchmark. `scripts/run_fdb.sh` invokes inference with `--copy-companions`, so
missing input/annotation files are copied without modifying the source dataset
or overwriting generated files.

The upstream ASR entry point used by the script is:

```bash
python /path/to/Full-Duplex-Bench/v1_v1.5/get_transcript/asr.py \
  --root_dir /path/to/results/v1_0/candor_pause_handling \
  --task default
```

For the v1.0 synthetic interruption task, use
`--task user_interruption`; all other tasks use `default`. The evaluator entry
point is exactly
`/path/to/Full-Duplex-Bench/v1_v1.5/evaluation/evaluate.py`.

Commands used for FDB v1.0:

```bash
python evaluate.py --task pause_handling --root_dir /path/to/candor_pause_handling
python evaluate.py --task pause_handling --root_dir /path/to/synthetic_pause_handling
python evaluate.py --task backchannel --root_dir /path/to/icc_backchannel
python evaluate.py --task smooth_turn_taking --root_dir /path/to/candor_turn_taking
python evaluate.py --task user_interruption --root_dir /path/to/synthetic_user_interruption
```

Commands used for each FDB v1.5 task (`background_speech`,
`talking_to_other`, `user_backchannel`, and `user_interruption`):

```bash
python get_timing.py --root_dir /path/to/task
python evaluate.py --task general_before_after --root_dir /path/to/task
```

The upstream code and evaluator remain subject to the upstream license.
`user_interruption` requires `OPENAI_API_KEY`, as documented upstream.
The end-to-end script skips only that evaluator with a clear message when the
key is absent; ASR and other metrics continue.

From the EmotionPlex repository root, the complete command is:

```bash
scripts/run_fdb.sh \
  /path/to/fdb/dataset \
  /path/to/results/fdb \
  /path/to/Full-Duplex-Bench \
  cuda:0 \
  - \
  v1_0 v1_5
```
