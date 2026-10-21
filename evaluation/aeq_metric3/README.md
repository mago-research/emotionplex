# AEQ Metric 3

Metric 3 measures whether the generated assistant response expresses the
requested emotion (`happy`, `sad`, `angry`, or `neutral`).

- `metric3_emotion2vec.py`: speech-emotion classification and accuracy.
- `metric3_text_judge.py`: OpenAI text-only 1–5 emotion score.
- `emotion2vec_utils.py`: shared emotion2vec model metadata and label helpers.

Generate the input with the reproducible
[`inference/aeq_metric3` runner](../../inference/aeq_metric3/README.md).

## Input CSV

Both commands consume the runner's `predictions.csv`. Required and optional
columns are:

- `requested_emotion` (required by both): one of the four target labels.
- `output_wav_path` (required for audio scoring): path to generated response
  audio. An empty value is counted as an audio miss.
- `response_text` (required for text scoring): generated assistant text. Empty
  values are skipped.
- `context`, `transcript` (optional): conversation context and user text sent
  to the text judge; missing values become empty strings.

Other columns are preserved unchanged in output CSVs.

## Run

From this directory:

```bash
python metric3_emotion2vec.py \
  --input-csv predictions.csv \
  --out-csv results/metric3_audio.csv \
  --device cuda:0
```

For the text judge, prefer an exported credential:

```bash
export OPENAI_API_KEY=...
python metric3_text_judge.py \
  --input-csv predictions.csv \
  --out-csv results/metric3_text.csv
```

Alternatively, copy `.env.example` to `.env` in this directory and fill it
locally. Existing process environment variables take precedence. `.env` is
ignored by Git and must not be committed.

Each command writes the scored CSV, a log beside it, and CSV/Markdown summary
files. Use `--help` for model, worker, output, and CPU-device options.

From the repository root, generation and both evaluators can be chained with:

```bash
scripts/run_aeq_metric3.sh \
  /path/to/aeq/dataset.json \
  /path/to/aeq/wavs \
  /path/to/results/aeq \
  cuda:0
```

Without `OPENAI_API_KEY`, this command explicitly skips the text judge after
completing inference and emotion2vec scoring.
