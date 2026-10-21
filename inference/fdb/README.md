# Full-Duplex-Bench inference

This runner applies one PersonaPlex model instance to all selected
Full-Duplex-Bench (FDB) samples. Obtain FDB v1.0/v1.5 from the
[official Full-Duplex-Bench repository](https://github.com/DanielLin94144/Full-Duplex-Bench)
and follow its dataset licenses and terms.

## Input and output

The dataset root must use this layout:

```text
dataset/
  <split>/
    <task>/
      <sample>/
        input.wav
```

The output root mirrors it and contains only generated artifacts:

```text
output/
  manifest.jsonl
  <split>/<task>/<sample>/
    output.wav
    output_tokens.json
```

`output_tokens.json` is the raw PersonaPlex text-token dump. It is deliberately
not named `output.json`: the upstream FDB ASR stage owns `output.json` and its
evaluators expect that file to contain an aligned ASR transcript.

## Run

Install the shared dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r inference/requirements.txt
```

Run all discovered splits with stock PersonaPlex:

```bash
python inference/fdb/run_fdb.py \
  --dataset /path/to/fdb/dataset \
  --output /path/to/fdb/personaplex-output \
  --copy-companions
```

Select splits or use a merged fine-tuned checkpoint:

```bash
python inference/fdb/run_fdb.py \
  --dataset /path/to/fdb/dataset \
  --output /path/to/fdb/personaplex-output \
  --splits v1_0 v1_5 \
  --model-weight /path/to/model.safetensors \
  --copy-companions \
  --resume
```

The preserved PersonaPlex FDB defaults are voice `NATF2.pt`, seed `42424242`,
audio/text temperatures `0.8`/`0.7`, and audio/text top-k `250`/`25`.
Interruption task names always receive the documented teacher prompt. Every
other task receives `You enjoy having a good conversation.`; use
`--conversation-prompt "Answer with a neutral tone."` for the neutral
alternative.

`--resume` skips a sample only if its manifest provenance matches the current
input metadata and prompt, and both `output.wav` and a valid string-list
`output_tokens.json` exist. A pre-existing output file without a matching
manifest record is regenerated. Without `--resume`, a non-empty output
directory is rejected.

`--copy-companions` copies missing `input.wav`, clean inputs, and annotation
JSON files from each dataset sample with `shutil.copy2`. It never modifies the
dataset and never overwrites existing destination files or the reserved
`output.wav`, `output.json`, and `output_tokens.json` names. Companion copying
runs before resume filtering, so it also repairs missing files for already
completed samples.

For the complete inference → upstream ASR → evaluation flow, use
`scripts/run_fdb.sh`; it always enables companion copying.
