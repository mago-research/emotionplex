#!/usr/bin/env python3
"""Generate the four prompted-tone responses used by AEQ Metric 3."""

from __future__ import annotations

import argparse
import csv
import json
import random
import string
import sys
import time
from pathlib import Path
from typing import Any


INFERENCE_DIR = Path(__file__).resolve().parents[1]
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

REQUESTED_EMOTIONS = ("happy", "sad", "angry", "neutral")
DEFAULT_PROMPT_TEMPLATE = "Answer with a {emotion} tone."
DEFAULT_HF_REPO = "nvidia/personaplex-7b-v1"
MIMI_SAMPLE_RATE = 24_000
SPECIAL_TOKENS = {"EPAD", "BOS", "EOS", "PAD"}
EMOTION_TO_WAV_PREFIXES = {
    "Angry": ("anger",),
    "Disgusted": ("disgust", "Disgust"),
    "Amused": ("amused",),
    "Neutral": ("neutral", "Neutral"),
}
FIELDNAMES = [
    "sentence_group_id",
    "transcript",
    "speaker",
    "gt_emotion",
    "utt_id",
    "input_wav_path",
    "output_wav_path",
    "response_text",
    "generate_time_s",
    "error",
    "context",
    "gt_explanation",
    "gt_response",
    "requested_emotion",
    "text_prompt",
]


def validate_prompt_template(template: str) -> None:
    fields = [
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name is not None
    ]
    if "emotion" not in fields:
        raise ValueError("Prompt template must contain the {emotion} field")
    unexpected = sorted(set(fields) - {"emotion"})
    if unexpected:
        raise ValueError(
            "Prompt template has unsupported fields: " + ", ".join(unexpected)
        )
    try:
        template.format(emotion="happy")
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid prompt template: {exc}") from exc


def find_wav(
    wav_files: list[Path], index_suffix: str, prefixes: tuple[str, ...]
) -> Path | None:
    """Find the deterministic first benchmark WAV matching emotion and ID."""
    for wav in wav_files:
        if wav.stem.endswith(f"_{index_suffix}") and any(
            wav.name.startswith(f"{prefix}_") for prefix in prefixes
        ):
            return wav
    return None


def _load_entries(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.is_file():
        raise ValueError(f"Dataset JSON not found: {json_path}")
    try:
        with json_path.open(encoding="utf-8") as handle:
            entries = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read dataset JSON {json_path}: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError("Dataset JSON must be a non-empty list")
    return entries


def build_metric3_jobs(
    json_path: Path,
    wav_dir: Path,
    limit: int | None,
    seed: int,
    prompt_template: str,
) -> list[dict[str, str]]:
    """Validate AEQ input and build four reproducible jobs per sentence."""
    validate_prompt_template(prompt_template)
    if not wav_dir.is_dir():
        raise ValueError(f"WAV directory not found: {wav_dir}")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be greater than zero")

    entries = _load_entries(json_path)
    if limit is not None:
        entries = entries[:limit]
    wav_files = sorted(
        path for path in wav_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not wav_files:
        raise ValueError(f"No .wav files found in {wav_dir}")

    rng = random.Random(seed)
    jobs: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset entry {position} is not an object")
        missing = [key for key in ("id", "text", "contexts") if key not in entry]
        if missing:
            raise ValueError(
                f"Dataset entry {position} is missing: {', '.join(missing)}"
            )
        entry_id = entry["id"]
        contexts = entry["contexts"]
        if not isinstance(entry_id, str) or len(entry_id) < 4:
            raise ValueError(f"Dataset entry {position} has an invalid string id")
        if entry_id in seen_ids:
            raise ValueError(f"Duplicate dataset id: {entry_id}")
        seen_ids.add(entry_id)
        if not isinstance(entry["text"], str) or not isinstance(contexts, dict):
            raise ValueError(
                f"Dataset entry {entry_id} needs string text and object contexts"
            )

        index_suffix = entry_id[-4:]
        candidates: list[tuple[str, Path]] = []
        for emotion, prefixes in EMOTION_TO_WAV_PREFIXES.items():
            wav = find_wav(wav_files, index_suffix, prefixes)
            if wav is not None:
                candidates.append((emotion, wav))
        if not candidates:
            raise ValueError(
                f"No matching WAV for dataset id {entry_id} "
                f"(expected filename suffix _{index_suffix}.wav)"
            )

        source_emotion, wav = rng.choice(candidates)
        for requested in REQUESTED_EMOTIONS:
            jobs.append(
                {
                    "sentence_group_id": entry_id,
                    "transcript": entry["text"],
                    "gt_emotion": source_emotion.lower(),
                    "utt_id": wav.stem,
                    "input_wav_path": str(wav.resolve()),
                    "context": str(contexts.get("Context", "")),
                    "gt_explanation": str(
                        contexts.get(f"{source_emotion}_explanation", "")
                    ),
                    "gt_response": str(
                        contexts.get(f"{source_emotion}_response", "")
                    ),
                    "requested_emotion": requested,
                    "text_prompt": prompt_template.format(emotion=requested),
                }
            )
    return jobs


def write_user_wav_with_hold(
    input_path: Path, output_path: Path, hold_seconds: float
) -> None:
    import numpy as np
    import sphn

    pcm, sample_rate = sphn.read(str(input_path))
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim == 1:
        user = pcm
    elif pcm.ndim == 2 and pcm.shape[0] <= 8 and pcm.shape[0] < pcm.shape[-1]:
        rms = np.sqrt(np.mean(pcm.astype(np.float64) ** 2, axis=1))
        user = pcm[int(np.argmax(rms))]
    elif pcm.ndim == 2:
        rms = np.sqrt(np.mean(pcm.astype(np.float64) ** 2, axis=0))
        user = pcm[:, int(np.argmax(rms))]
    else:
        raise ValueError(f"Unsupported WAV shape {pcm.shape}: {input_path}")
    if user.size == 0:
        raise ValueError(f"Empty WAV: {input_path}")

    if int(sample_rate) != MIMI_SAMPLE_RATE:
        resampled = sphn.resample(
            user[None, :],
            src_sample_rate=int(sample_rate),
            dst_sample_rate=MIMI_SAMPLE_RATE,
        )
        user = np.asarray(resampled, dtype=np.float32).reshape(-1)
    hold_samples = int(round(hold_seconds * MIMI_SAMPLE_RATE))
    padded = np.concatenate(
        [user, np.zeros(hold_samples, dtype=np.float32)]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(output_path), padded, MIMI_SAMPLE_RATE)


def response_text_from_dump(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        tokens = json.load(handle)
    if not isinstance(tokens, list) or not all(
        isinstance(token, str) for token in tokens
    ):
        raise ValueError(f"Generated token dump is not a string list: {path}")
    return "".join(token for token in tokens if token not in SPECIAL_TOKENS).strip()


def completed_keys(predictions_csv: Path) -> set[tuple[str, str]]:
    """Read resume keys and reject ambiguous or incompatible CSV files."""
    if not predictions_csv.exists():
        return set()
    with predictions_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"utt_id", "requested_emotion"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{predictions_csv} must contain utt_id and requested_emotion"
            )
        keys: set[tuple[str, str]] = set()
        for line_number, row in enumerate(reader, start=2):
            key = (
                (row.get("utt_id") or "").strip(),
                (row.get("requested_emotion") or "").strip().lower(),
            )
            if not all(key):
                raise ValueError(
                    f"Missing resume key in {predictions_csv}:{line_number}"
                )
            if key in keys:
                raise ValueError(
                    f"Duplicate resume key {key} in {predictions_csv}"
                )
            keys.add(key)
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True, help="AEQ dataset JSON")
    parser.add_argument(
        "--wav-dir", type=Path, required=True, help="Directory of AEQ input WAVs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory, or an existing run with --skip-existing",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--voice-prompt", default="NATF0.pt")
    parser.add_argument("--voice-prompt-dir", type=Path)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument(
        "--moshi-weight",
        type=Path,
        help="Merged fine-tuned model.safetensors; omit for stock PersonaPlex",
    )
    parser.add_argument("--mimi-weight", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument(
        "--pad-seconds",
        "--hold-seconds",
        dest="hold_seconds",
        type=float,
        default=10.0,
        help="Trailing silent hold in seconds (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42_424_242)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--temp-audio", type=float, default=0.8)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--topk-audio", type=int, default=250)
    parser.add_argument("--topk-text", type=int, default=25)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    validate_prompt_template(args.prompt_template)
    if args.hold_seconds < 0:
        raise ValueError("--hold-seconds must be non-negative")
    for name in ("temp_audio", "temp_text"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("topk_audio", "topk_text"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for optional_file in ("moshi_weight", "mimi_weight", "tokenizer"):
        value = getattr(args, optional_file)
        if value is not None and not value.is_file():
            raise ValueError(
                f"--{optional_file.replace('_', '-')} not found: {value}"
            )
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError(f"--output-dir is not a directory: {args.output_dir}")
    output = args.output_dir.resolve()
    if output in {args.json.resolve(), args.wav_dir.resolve()}:
        raise ValueError("--output-dir must differ from input paths")


def main() -> None:
    args = parse_args()
    try:
        _validate_args(args)
        jobs = build_metric3_jobs(
            args.json,
            args.wav_dir,
            args.limit,
            args.seed,
            args.prompt_template,
        )
    except ValueError as exc:
        raise SystemExit(f"Input error: {exc}") from exc

    predictions_csv = args.output_dir / "predictions.csv"
    if predictions_csv.exists() and not args.skip_existing:
        raise SystemExit(
            f"{predictions_csv} already exists; choose a new --output-dir or "
            "pass --skip-existing to resume"
        )
    try:
        done = completed_keys(predictions_csv) if args.skip_existing else set()
    except ValueError as exc:
        raise SystemExit(f"Resume error: {exc}") from exc
    pending = [
        job
        for job in jobs
        if (job["utt_id"], job["requested_emotion"]) not in done
    ]
    print(
        f"[System] {len(jobs)} total job(s); {len(done)} CSV row(s) already "
        f"present; {len(pending)} pending"
    )
    if not pending:
        print(f"[System] Nothing to do: {predictions_csv}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = args.output_dir / "wavs"
    texts_dir = args.output_dir / "texts"
    padded_dir = args.output_dir / "_padded_inputs"
    for directory in (wavs_dir, texts_dir, padded_dir):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        from personaplex_backend import (
            BackendConfig,
            PersonaPlexBackend,
            resolve_voice_prompt,
        )

        voice_prompt = resolve_voice_prompt(
            args.voice_prompt, args.voice_prompt_dir, args.hf_repo
        )
        evaluator = PersonaPlexBackend(
            BackendConfig(
                hf_repo=args.hf_repo,
                moshi_weight=args.moshi_weight,
                mimi_weight=args.mimi_weight,
                tokenizer=args.tokenizer,
                device=args.device,
                seed=args.seed,
                cpu_offload=args.cpu_offload,
                greedy=args.greedy,
                temp_audio=args.temp_audio,
                temp_text=args.temp_text,
                topk_audio=args.topk_audio,
                topk_text=args.topk_text,
            ),
            voice_prompt,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"Backend error: {exc}") from exc

    write_header = not predictions_csv.exists()
    padded_done: set[str] = set()
    ok = 0
    failed = 0
    with predictions_csv.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            csv_file.flush()

        for index, job in enumerate(pending, start=1):
            basename = f"{job['utt_id']}_{job['requested_emotion']}_personaplex"
            out_wav = wavs_dir / f"{basename}.wav"
            out_json = texts_dir / f"{basename}.json"
            padded_input = padded_dir / f"{job['utt_id']}_padded.wav"
            start = time.perf_counter()
            error = ""
            response_text = ""
            success = False
            try:
                if job["utt_id"] not in padded_done:
                    write_user_wav_with_hold(
                        Path(job["input_wav_path"]),
                        padded_input,
                        args.hold_seconds,
                    )
                    padded_done.add(job["utt_id"])
                agent_pcm = evaluator.generate(
                    padded_input, job["text_prompt"], out_wav, out_json
                )
                response_text = response_text_from_dump(out_json)
                success = True
                ok += 1
                try:
                    import numpy as np

                    peak = (
                        float(np.max(np.abs(agent_pcm))) if agent_pcm.size else 0.0
                    )
                    if peak < 0.01:
                        print(f"    [Warn] output peak {peak:.5f}; audio may be silent")
                except (AttributeError, TypeError):
                    pass
            except Exception as exc:  # Keep one auditable CSV row per requested job.
                error = f"{type(exc).__name__}: {exc}"
                failed += 1

            elapsed = time.perf_counter() - start
            writer.writerow(
                {
                    "sentence_group_id": job["sentence_group_id"],
                    "transcript": job["transcript"],
                    "speaker": "emovdb",
                    "gt_emotion": job["gt_emotion"],
                    "utt_id": job["utt_id"],
                    "input_wav_path": job["input_wav_path"],
                    "output_wav_path": str(out_wav.resolve()) if success else "",
                    "response_text": response_text,
                    "generate_time_s": f"{elapsed:.2f}",
                    "error": error,
                    "context": job["context"],
                    "gt_explanation": job["gt_explanation"],
                    "gt_response": job["gt_response"],
                    "requested_emotion": job["requested_emotion"],
                    "text_prompt": job["text_prompt"],
                }
            )
            csv_file.flush()
            status = "ok" if success else f"FAILED: {error}"
            print(
                f"[{index}/{len(pending)}] {job['utt_id']} / "
                f"{job['requested_emotion']}: {status}"
            )

    print(f"[System] Done: {ok} ok, {failed} failed. Wrote {predictions_csv}")


if __name__ == "__main__":
    main()
