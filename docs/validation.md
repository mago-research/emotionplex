# Inference validation

## Automated checks

The public inference pipeline is covered by CPU-only checks that do not
download model weights:

```bash
python -m unittest discover -s tests -v
python -m compileall -q inference evaluation tests
bash -n scripts/run_aeq_metric3.sh scripts/run_fdb.sh
python inference/aeq_metric3/run_batch_metric3.py --help
python inference/fdb/run_fdb.py --help
```

These checks cover AEQ four-emotion job construction and CSV schema, FDB
prompt selection and dataset discovery, manifest-based resume behavior, and
non-destructive companion-file copying.

## GPU smoke matrix

Before a release, run one AEQ sentence and one FDB sample with both model
conditions:

- Stock `nvidia/personaplex-7b-v1`: AEQ generation, FDB generation, and
  evaluator connection are pending a GPU environment.
- EmotionPlex merged checkpoint: AEQ generation, FDB generation, and evaluator
  connection are pending a GPU environment.

The current Cursor execution environment exposes the host NVIDIA driver but
does not expose CUDA device nodes, so model generation cannot run in this
session. This is an infrastructure limitation, not a successful model smoke
test.

Run the stock checks from the repository root:

```bash
PERSONAPLEX_PYTHON=/path/to/personaplex/bin/python \
AEQ_EVAL_PYTHON=/path/to/aeq-eval/bin/python \
scripts/run_aeq_metric3.sh \
  /path/to/aeq/dataset.json \
  /path/to/aeq/wavs \
  /path/to/results/aeq-stock \
  cuda:0

PERSONAPLEX_PYTHON=/path/to/personaplex/bin/python \
FDB_PYTHON=/path/to/fdb-eval/bin/python \
scripts/run_fdb.sh \
  /path/to/fdb/dataset \
  /path/to/results/fdb-stock \
  /path/to/Full-Duplex-Bench \
  cuda:0 \
  - \
  v1_0
```

Repeat both commands with the merged EmotionPlex `model.safetensors` as the
last AEQ argument and in place of `-` in the FDB command. For a bounded smoke
test, invoke the Python runners directly with `--limit 1` before running the
evaluators.

Successful validation requires:

- non-empty generated WAV files;
- parseable token JSON lists;
- one complete AEQ `predictions.csv` row per requested emotion;
- successful AEQ emotion2vec scoring;
- FDB output under `<split>/<task>/<sample>/`;
- upstream ASR-generated `output.json` and a task evaluator result;
- no model weights, credentials, audio, or result files committed to Git.
