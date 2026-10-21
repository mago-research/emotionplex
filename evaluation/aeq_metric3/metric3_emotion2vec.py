#!/usr/bin/env python3
"""Metric 3 audio score: emotion2vec accuracy vs the *requested* prompt emotion.

Unlike Metric 1, this does not use ALLOWED_RESPONSE_EMOTIONS (empathetic
appropriateness). The target is always the `requested_emotion` input column
(happy/sad/angry/neutral). Empty/failed generations
count as misses.

emotion2vec's extra labels are collapsed onto the 4-class SFT set for
matching only (raw top-1 is still stored in speech_emotion):
  disgusted -> angry, fearful -> sad.
surprised / other / unk are left unmatched.

Input: a predictions.csv (needs output_wav_path +
requested_emotion).
Output:
  1. --out-csv: same rows + speech_emotion, speech_emotion_collapsed,
     speech_emotion_top1_score, audio_match, per-label raw scores.
  2. --summary-csv / --summary-md: accuracy overall and per requested
     emotion, predicted-label distribution, and requested-vs-predicted
     confusion (raw emotion2vec labels).
"""

import argparse
import csv
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from emotion2vec_utils import MODEL_ID, MODEL_LABELS_EN, MODEL_REVISION, label_en

REQUESTED_EMOTIONS = ["happy", "sad", "angry", "neutral"]
NONE_LABEL = "none"

# Collapse emotion2vec labels that the 4-class SFT model does not produce
# onto the nearest SFT class. Used only for audio_match; speech_emotion
# stays the raw emotion2vec top-1. surprised/other/unk are too ambiguous
# (or non-emotions) to fold in.
SFT_COLLAPSE = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "disgusted": "angry",
    "fearful": "sad",
}


def collapse_to_sft(label: str) -> str:
    """Map an emotion2vec label onto the 4-class SFT set, or '' if unmatched."""
    return SFT_COLLAPSE.get((label or "").strip().lower(), "")


def _requested(row: dict) -> str:
    return (row.get("requested_emotion") or "").strip().lower()


def _pred(row: dict) -> str:
    return (row.get("speech_emotion") or "").strip().lower() or NONE_LABEL


def _collapsed(row: dict) -> str:
    raw = _pred(row)
    if raw == NONE_LABEL:
        return ""
    return collapse_to_sft(raw)


def build_audio_summary(rows: list[dict]) -> list[dict]:
    """One row per requested emotion + Overall: n, match count/rate
    (failures count as misses), predicted-label distribution, and
    confusion counts against every emotion2vec label plus `none`."""
    by_req: dict[str, list[dict]] = {}
    for r in rows:
        by_req.setdefault(_requested(r), []).append(r)

    dist_labels = MODEL_LABELS_EN + [NONE_LABEL]
    ordered = [e for e in REQUESTED_EMOTIONS if e in by_req] + sorted(
        e for e in by_req if e not in REQUESTED_EMOTIONS
    )

    summary_rows = []
    overall_counter: Counter = Counter()
    overall_n = 0
    overall_match = 0

    def make_row(name: str, group: list[dict], counter: Counter, match_count: int) -> dict:
        n = len(group)
        row = {
            "requested_emotion": name,
            "n": n,
            "match_count": match_count,
            "accuracy_pct": round(100 * match_count / n, 1) if n else "",
        }
        for l in dist_labels:
            row[f"count_{l}"] = counter.get(l, 0)
            row[f"pct_{l}"] = round(100 * counter.get(l, 0) / n, 1) if n else 0.0
        return row

    for emo in ordered:
        group = by_req[emo]
        counter = Counter(_pred(r) for r in group)
        match_count = sum(1 for r in group if _collapsed(r) == emo)
        summary_rows.append(make_row(emo, group, counter, match_count))
        overall_counter.update(counter)
        overall_n += len(group)
        overall_match += match_count

    summary_rows.append(make_row("Overall", rows, overall_counter, overall_match))
    return summary_rows


def write_summary_csv(summary_rows: list[dict], path: Path):
    dist_labels = MODEL_LABELS_EN + [NONE_LABEL]
    fieldnames = (
        ["requested_emotion", "n", "match_count", "accuracy_pct"]
        + [f"count_{l}" for l in dist_labels]
        + [f"pct_{l}" for l in dist_labels]
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_summary_md(summary_rows: list[dict], path: Path):
    dist_labels = MODEL_LABELS_EN + [NONE_LABEL]
    lines = [
        "| requested emotion | n | match | dist(speech_emotion) |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        dist_items = sorted(
            ((l, row[f"count_{l}"]) for l in dist_labels if row[f"count_{l}"] > 0),
            key=lambda kv: -kv[1],
        )
        dist_str = ", ".join(f"{l} {n} ({row[f'pct_{l}']}%)" for l, n in dist_items) or "-"
        acc = row["accuracy_pct"]
        acc_str = f"{row['match_count']}/{row['n']} ({acc}%)" if acc != "" else "-"
        lines.append(
            f"| `{row['requested_emotion']}` | {row['n']} | {acc_str} | {dist_str} |"
        )

    lines.extend([
        "",
        "Confusion (requested \\ predicted):",
        "",
        "| requested \\ predicted | " + " | ".join(f"`{l}`" for l in dist_labels) + " |",
        "| --- | " + " | ".join("---" for _ in dist_labels) + " |",
    ])
    for row in summary_rows:
        if row["requested_emotion"] == "Overall":
            continue
        cells = " | ".join(str(row[f"count_{l}"]) for l in dist_labels)
        lines.append(f"| `{row['requested_emotion']}` | {cells} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True, help="predictions.csv")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None,
                         help="per-requested-emotion accuracy + confusion (default: <out-csv stem>_summary.csv)")
    parser.add_argument("--summary-md", type=Path, default=None,
                         help="same summary as a Markdown table (default: <out-csv stem>_summary.md)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="FunASR device (for example cuda:0 or cpu)",
    )
    args = parser.parse_args()
    if args.summary_csv is None:
        args.summary_csv = args.out_csv.with_name(args.out_csv.stem + "_summary.csv")
    if args.summary_md is None:
        args.summary_md = args.out_csv.with_name(args.out_csv.stem + "_summary.md")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out_csv.with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("metric3_emotion2vec")

    rows = list(csv.DictReader(open(args.input_csv, encoding="utf-8")))
    if not rows:
        log.error("Empty CSV: %s", args.input_csv)
        sys.exit(1)
    if "requested_emotion" not in rows[0]:
        log.error("%s has no requested_emotion column", args.input_csv)
        sys.exit(1)

    scorable_rows = [r for r in rows if r.get("output_wav_path")]
    log.info("Loaded %d rows (%d with a generated output wav) from %s", len(rows), len(scorable_rows), args.input_csv)

    log.info("Loading model %s ...", MODEL_ID)
    from funasr import AutoModel
    model = AutoModel(
        model=MODEL_ID, model_revision=MODEL_REVISION,
        disable_update=True, disable_pbar=True, device=args.device,
    )
    log.info("Model loaded.")

    extra = ["speech_emotion", "speech_emotion_collapsed", "speech_emotion_top1_score", "audio_match"] + [f"speech_score_{l}" for l in MODEL_LABELS_EN]
    out_fields = list(rows[0].keys()) + extra
    n_error = 0
    t_start = time.time()
    out_rows: list[dict] = []

    def blank_scores(out_row: dict):
        out_row["speech_emotion"] = ""
        out_row["speech_emotion_collapsed"] = ""
        out_row["speech_emotion_top1_score"] = ""
        out_row["audio_match"] = 0
        for l in MODEL_LABELS_EN:
            out_row[f"speech_score_{l}"] = ""

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            out_row = dict(row)
            requested = _requested(row)
            if not row.get("output_wav_path"):
                blank_scores(out_row)
                writer.writerow(out_row)
                out_rows.append(out_row)
                continue
            try:
                res = model.generate(row["output_wav_path"], output_dir=None, granularity="utterance", extract_embedding=False)
                item = res[0]
                labels_raw = item["labels"]
                scores = item["scores"]
                top_idx = max(range(len(scores)), key=lambda i: scores[i])
                pred_en = label_en(labels_raw[top_idx])
                out_row["speech_emotion"] = pred_en
                collapsed = collapse_to_sft(pred_en)
                out_row["speech_emotion_collapsed"] = collapsed
                out_row["speech_emotion_top1_score"] = scores[top_idx]
                out_row["audio_match"] = int(collapsed == requested)
                score_by_label = {label_en(l): s for l, s in zip(labels_raw, scores)}
                for l in MODEL_LABELS_EN:
                    out_row[f"speech_score_{l}"] = score_by_label.get(l, "")
            except Exception:
                log.exception("SER failed for %s", row["output_wav_path"])
                n_error += 1
                blank_scores(out_row)
            writer.writerow(out_row)
            out_rows.append(out_row)

            if i % 50 == 0 or i == len(rows):
                f.flush()
                log.info("%d/%d done, elapsed=%.1fs, errors=%d", i, len(rows), time.time() - t_start, n_error)

    log.info("DONE. total=%d errors=%d elapsed=%.1fs", len(rows), n_error, time.time() - t_start)
    log.info("Wrote %s", args.out_csv)

    summary_rows = build_audio_summary(out_rows)
    write_summary_csv(summary_rows, args.summary_csv)
    write_summary_md(summary_rows, args.summary_md)
    log.info("Wrote %s", args.summary_csv)
    log.info("Wrote %s", args.summary_md)
    for row in summary_rows:
        log.info(
            "  %-10s n=%-4d match=%s/%s acc=%s",
            row["requested_emotion"], row["n"],
            row["match_count"], row["n"], row["accuracy_pct"],
        )


if __name__ == "__main__":
    main()
