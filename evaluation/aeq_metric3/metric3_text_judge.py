#!/usr/bin/env python3
"""Metric 3 text score: GPT-4o 1-5 how well the *written* reply expresses
the requested prompt emotion.

Pass the whole conversation (context, user transcript, assistant text)
plus the expected emotion. Scores only text emotion, not spoken delivery
and not empathy. Empty response_text is not sent to the judge (same skip
convention as Metric 2-1).

Input: a predictions.csv (needs response_text +
requested_emotion).
Output: same rows + text_judge_raw, text_score (1-5, or blank if skipped
or the judge failed), plus a per-requested-emotion summary.
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

# The process environment takes precedence. A local, untracked .env is optional.
_local_env = Path(__file__).resolve().parent / ".env"
if _local_env.is_file():
    load_dotenv(_local_env, override=False)

REQUESTED_EMOTIONS = ["happy", "sad", "angry", "neutral"]

JUDGE_SYSTEM_PROMPT = """You are scoring whether the assistant's written reply expresses a requested emotion. Ignore spoken delivery, tone of voice, and whether the reply is empathetic or on-topic. Score only the emotion expressed in the assistant's text.

Scoring rubric:
1 - The text's emotion is completely wrong or absent.
2 - The text's emotion does not match.
3 - Slight mismatch; some of the expected emotion is present.
4 - The text's emotion is generally the expected one.
5 - The text clearly and accurately expresses the expected emotion.

Output only a single Arabic numeral from 1 to 5. No explanation, no other text."""

USER_TEMPLATE = """[Context]
{context}

[User]
{transcript}

[Assistant]
{response}

[Expected emotion]
{requested_emotion}"""

SCORE_RE = re.compile(r"([1-5])")


def judge_one(client, model, context, transcript, response, requested_emotion):
    user_content = USER_TEMPLATE.format(
        context=context or "(none)",
        transcript=transcript,
        response=response,
        requested_emotion=requested_emotion,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        seed=1337,
    )
    text = resp.choices[0].message.content.strip()
    m = SCORE_RE.search(text)
    score = int(m.group(1)) if m else None
    return text, score


def _requested(row: dict) -> str:
    return (row.get("requested_emotion") or "").strip().lower()


def build_text_summary(rows: list[dict]) -> list[dict]:
    by_req: dict[str, list[dict]] = {}
    for r in rows:
        by_req.setdefault(_requested(r), []).append(r)

    ordered = [e for e in REQUESTED_EMOTIONS if e in by_req] + sorted(
        e for e in by_req if e not in REQUESTED_EMOTIONS
    )

    def make_row(name: str, group: list[dict]) -> dict:
        scores = []
        for r in group:
            raw = r.get("text_score", "")
            if raw == "" or raw is None:
                continue
            try:
                scores.append(int(raw))
            except (TypeError, ValueError):
                continue
        dist = Counter(scores)
        n = len(group)
        n_scored = len(scores)
        row = {
            "requested_emotion": name,
            "n": n,
            "n_scored": n_scored,
            "avg_score": round(sum(scores) / n_scored, 2) if n_scored else "",
        }
        for s in range(1, 6):
            row[f"count_{s}"] = dist.get(s, 0)
        return row

    summary_rows = [make_row(emo, by_req[emo]) for emo in ordered]
    summary_rows.append(make_row("Overall", rows))
    return summary_rows


def write_summary_csv(summary_rows: list[dict], path: Path):
    fieldnames = ["requested_emotion", "n", "n_scored", "avg_score"] + [f"count_{s}" for s in range(1, 6)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_summary_md(summary_rows: list[dict], path: Path):
    lines = [
        "| requested emotion | n | n_scored | avg score | score distribution (1–5) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        dist = ", ".join(f"{s}: {row[f'count_{s}']}" for s in range(1, 6) if row[f"count_{s}"]) or "-"
        avg = row["avg_score"] if row["avg_score"] != "" else "-"
        lines.append(
            f"| `{row['requested_emotion']}` | {row['n']} | {row['n_scored']} | {avg} | {dist} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True, help="predictions.csv")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.summary_csv is None:
        args.summary_csv = args.out_csv.with_name(args.out_csv.stem + "_summary.csv")
    if args.summary_md is None:
        args.summary_md = args.out_csv.with_name(args.out_csv.stem + "_summary.md")
    if not args.input_csv.is_file():
        parser.error(
            f"input CSV not found: {args.input_csv}. Generate predictions.csv with "
            "inference/aeq_metric3/run_batch_metric3.py"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out_csv.with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("metric3_text_judge")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set.")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    with args.input_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        log.error("Empty CSV: %s", args.input_csv)
        sys.exit(1)
    if "requested_emotion" not in rows[0]:
        log.error(
            "%s has no requested_emotion column; pass predictions.csv from "
            "inference/aeq_metric3/run_batch_metric3.py",
            args.input_csv,
        )
        sys.exit(1)

    scorable_idx = [i for i, r in enumerate(rows) if r.get("response_text")]
    log.info("Loaded %d rows (%d judgeable: has response_text) from %s",
             len(rows), len(scorable_idx), args.input_csv)

    results = {}
    n_done = 0
    n_error = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i in scorable_idx:
            r = rows[i]
            fut = pool.submit(
                judge_one, client, args.model,
                r.get("context", ""), r.get("transcript", ""),
                r["response_text"], _requested(r),
            )
            futures[fut] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                raw_text, score = fut.result()
                results[i] = (raw_text, score)
            except Exception as e:
                log.exception("Judge failed for row %d", i)
                results[i] = (f"ERROR: {e!r}", None)
                n_error += 1
            n_done += 1
            if n_done % 50 == 0 or n_done == len(scorable_idx):
                log.info("%d/%d judged, elapsed=%.1fs, errors=%d", n_done, len(scorable_idx), time.time() - t_start, n_error)

    out_fields = list(rows[0].keys()) + ["text_judge_raw", "text_score"]
    out_rows = []
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for i, r in enumerate(rows):
            out_row = dict(r)
            if i in results:
                raw_text, score = results[i]
                out_row["text_judge_raw"] = raw_text
                out_row["text_score"] = score if score is not None else ""
            else:
                out_row["text_judge_raw"] = ""
                out_row["text_score"] = ""
            writer.writerow(out_row)
            out_rows.append(out_row)

    scores = [s for _, s in results.values() if s is not None]
    avg = sum(scores) / len(scores) if scores else None
    log.info("DONE. judged=%d errors=%d avg_score=%s elapsed=%.1fs", len(scorable_idx), n_error, avg, time.time() - t_start)
    log.info("Wrote %s", args.out_csv)

    summary_rows = build_text_summary(out_rows)
    write_summary_csv(summary_rows, args.summary_csv)
    write_summary_md(summary_rows, args.summary_md)
    log.info("Wrote %s", args.summary_csv)
    log.info("Wrote %s", args.summary_md)
    for row in summary_rows:
        log.info(
            "  %-10s n=%-4d n_scored=%-4d avg=%s",
            row["requested_emotion"], row["n"], row["n_scored"], row["avg_score"],
        )


if __name__ == "__main__":
    main()
