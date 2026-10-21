# AEQ Metric 3 inference

This runner reproduces the prompted PersonaPlex generation stage for AEQ
Metric 3. For each dataset sentence it deterministically selects one available
acted input take (seed `42424242`) and generates responses requested as
`happy`, `sad`, `angry`, and `neutral`.

The shared [`../personaplex_backend.py`](../personaplex_backend.py) backend is
a minimal adaptation of the generation loop in
`personaplex-finetune/pipeline/eval_emotion_tones.py`: it warms up Mimi and
Moshi once, loads the voice prompt once, then resets streaming state for each
job. The same loaded backend is reused by the FDB runner, and neither runner
requires that external repository at runtime.

## Requirements and model terms

Install from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r inference/requirements.txt
```

The inference requirements pin the
[NVIDIA/PersonaPlex](https://github.com/NVIDIA/personaplex) Moshi package at
commit `3428dfd95309a7f3c84fd93259ded0f810d1ff91`, which provides `LMGen` and
the required `moshi.offline` helpers. Install the system Opus development
package first if required by your platform. The stock model is downloaded from
[`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1);
review and comply with the model card and NVIDIA model license before use.
Weights are never stored in this repository.

## Run

Obtain the benchmark JSON and acted WAV files from an AEQ-Bench release; the
data is not redistributed here.

The dataset JSON, benchmark WAV directory, and output directory are explicit,
required inputs:

```bash
python inference/aeq_metric3/run_batch_metric3.py \
  --json /path/to/emotional_dialogues_bench_1context_gpt4.json \
  --wav-dir /path/to/emov_db_bench \
  --output-dir /path/to/run/metric3 \
  --voice-prompt NATF0.pt
```

`NATF0.pt` is resolved with PersonaPlex's Hugging Face voice-prompt helper.
Alternatively, pass an existing prompt file directly:

```bash
python inference/aeq_metric3/run_batch_metric3.py \
  --json /path/to/dataset.json \
  --wav-dir /path/to/wavs \
  --output-dir /path/to/run/metric3 \
  --voice-prompt /path/to/voice_prompt.pt \
  --moshi-weight /path/to/merged/model.safetensors
```

Omit `--moshi-weight` for stock PersonaPlex; the backend then downloads
`loaders.MOSHI_NAME` from `--hf-repo`. A merged fine-tuned checkpoint must be
provided with `--moshi-weight`. The preserved experimental defaults are:

- prompt `Answer with a {emotion} tone.`
- 10-second trailing hold at 24 kHz
- audio/text temperatures `0.8` / `0.7`
- audio/text top-k `250` / `25`
- seed `42424242` and voice prompt `NATF0.pt`

Outputs include `wavs/`, `texts/`, padded temporary inputs, and
`predictions.csv` with the schema consumed by `evaluation/aeq_metric3`.
`--skip-existing` resumes from `(utt_id, requested_emotion)` keys already in
the CSV. An output WAV by itself is not treated as completed; this prevents
missing CSV rows. Duplicate keys in an existing CSV are rejected.

To run generation followed by the audio evaluator and optional OpenAI text
judge:

```bash
scripts/run_aeq_metric3.sh \
  /path/to/aeq/dataset.json \
  /path/to/aeq/wavs \
  /path/to/results/aeq \
  cuda:0
```

The text judge is skipped with an explicit message unless `OPENAI_API_KEY` is
set.
