# Inference

Reproducible inference runners live here. Install their dependencies in a
separate Python 3.11 environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r inference/requirements.txt
```

AEQ Metric 3 generation is documented in
[`aeq_metric3/README.md`](aeq_metric3/README.md). Full-Duplex-Bench generation
is documented in [`fdb/README.md`](fdb/README.md). Both runners use the shared
`personaplex_backend.py`, which loads the model and voice prompt once per run
and resets streaming state between samples.

Model weights, datasets, voice prompts, and generated files are not part of
this repository.

## End-to-end commands

Run AEQ generation, emotion2vec scoring, and (when `OPENAI_API_KEY` is set)
the text judge:

```bash
scripts/run_aeq_metric3.sh \
  /path/to/aeq/dataset.json \
  /path/to/aeq/wavs \
  /path/to/results/aeq \
  cuda:0
```

Run FDB generation, companion copying, pinned-upstream ASR, and evaluation:

```bash
scripts/run_fdb.sh \
  /path/to/fdb/dataset \
  /path/to/results/fdb \
  /path/to/Full-Duplex-Bench \
  cuda:0 \
  - \
  v1_0 v1_5
```

The final `-` selects stock PersonaPlex; replace it with a merged
`model.safetensors` path. See each runner README and `script --help`/usage
output for details.

Inference and evaluation often require different environments. Set
`PERSONAPLEX_PYTHON` to the PersonaPlex environment, `AEQ_EVAL_PYTHON` to the
emotion2vec/OpenAI environment, and `FDB_PYTHON` to the upstream FDB
environment. Each variable defaults to `python`.

Release checks and the GPU smoke-test matrix are documented in
[`../docs/validation.md`](../docs/validation.md).
