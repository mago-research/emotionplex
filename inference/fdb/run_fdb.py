#!/usr/bin/env python3
"""Run PersonaPlex over a Full-Duplex-Bench dataset tree."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INFERENCE_DIR = Path(__file__).resolve().parents[1]
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

DEFAULT_HF_REPO = "nvidia/personaplex-7b-v1"
DEFAULT_CONVERSATION_PROMPT = "You enjoy having a good conversation."
INTERRUPTION_PROMPT = (
    "You are a wise and friendly teacher. Answer questions or provide advice "
    "in a clear and engaging way."
)
MANIFEST_NAME = "manifest.jsonl"
GENERATED_FILENAMES = {"output.wav", "output.json", "output_tokens.json"}


@dataclass(frozen=True)
class FDBJob:
    split: str
    task: str
    sample: str
    input_wav: Path
    output_dir: Path
    prompt: str

    @property
    def key(self) -> str:
        return f"{self.split}/{self.task}/{self.sample}"

    @property
    def output_wav(self) -> Path:
        return self.output_dir / "output.wav"

    @property
    def output_tokens(self) -> Path:
        return self.output_dir / "output_tokens.json"


def prompt_for_task(
    task: str, conversation_prompt: str = DEFAULT_CONVERSATION_PROMPT
) -> str:
    """Use the fixed teacher persona for every interruption task."""
    if "interruption" in task.casefold():
        return INTERRUPTION_PROMPT
    return conversation_prompt


def collect_fdb_jobs(
    dataset_root: Path,
    output_root: Path,
    splits: set[str] | None = None,
    conversation_prompt: str = DEFAULT_CONVERSATION_PROMPT,
) -> list[FDBJob]:
    """Collect ``<split>/<task>/<sample>/input.wav`` in stable order."""
    if not dataset_root.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_root}")

    available_splits = {path.name for path in dataset_root.iterdir() if path.is_dir()}
    if splits:
        missing = sorted(splits - available_splits)
        if missing:
            raise ValueError("Requested split(s) not found: " + ", ".join(missing))

    jobs: list[FDBJob] = []
    for split_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if splits and split_dir.name not in splits:
            continue
        for task_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for sample_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
                input_wav = sample_dir / "input.wav"
                if not input_wav.is_file():
                    continue
                jobs.append(
                    FDBJob(
                        split=split_dir.name,
                        task=task_dir.name,
                        sample=sample_dir.name,
                        input_wav=input_wav.resolve(),
                        output_dir=(
                            output_root
                            / split_dir.name
                            / task_dir.name
                            / sample_dir.name
                        ),
                        prompt=prompt_for_task(task_dir.name, conversation_prompt),
                    )
                )
    if not jobs:
        raise ValueError(
            f"No <split>/<task>/<sample>/input.wav files found in {dataset_root}"
        )
    return jobs


def copy_companion_files(job: FDBJob) -> int:
    """Copy missing dataset files beside outputs without touching either source.

    Generated output names are always reserved, even if unexpectedly present
    in the dataset sample directory.
    """
    copied = 0
    job.output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(job.input_wav.parent.iterdir()):
        if not source.is_file() or source.name in GENERATED_FILENAMES:
            continue
        destination = job.output_dir / source.name
        if destination.exists() or destination.is_symlink():
            continue
        shutil.copy2(source, destination)
        copied += 1
    return copied


def _job_signature(job: FDBJob) -> dict[str, Any]:
    stat = job.input_wav.stat()
    return {
        "key": job.key,
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "prompt": job.prompt,
        "output_wav": f"{job.key}/output.wav",
        "output_tokens": f"{job.key}/output_tokens.json",
    }


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load completed entries, rejecting malformed or duplicate records."""
    if not path.exists():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed manifest record at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
                raise ValueError(f"Invalid manifest record at {path}:{line_number}")
            key = entry["key"]
            if key in entries:
                raise ValueError(f"Duplicate manifest key {key!r} at {path}:{line_number}")
            entries[key] = entry
    return entries


def _valid_token_dump(path: Path) -> bool:
    try:
        tokens = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(tokens, list) and all(isinstance(token, str) for token in tokens)


def job_is_complete(job: FDBJob, manifest: dict[str, dict[str, Any]]) -> bool:
    """Require matching provenance plus both valid output files."""
    entry = manifest.get(job.key)
    if entry is None or entry != _job_signature(job):
        return False
    return (
        job.output_wav.is_file()
        and job.output_wav.stat().st_size > 0
        and job.output_tokens.is_file()
        and _valid_token_dump(job.output_tokens)
    )


def record_completion(
    path: Path,
    manifest: dict[str, dict[str, Any]],
    entry: dict[str, Any],
) -> None:
    """Atomically add or replace one completed job record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = {**manifest, entry["key"]: entry}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(updated):
            handle.write(
                json.dumps(updated[key], ensure_ascii=False, sort_keys=True) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    manifest.clear()
    manifest.update(updated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="FDB dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Output root")
    parser.add_argument(
        "--model-weight",
        type=Path,
        help="Merged model.safetensors; omit to use stock PersonaPlex",
    )
    parser.add_argument("--splits", nargs="+", help="Split names, for example v1_0 v1_5")
    parser.add_argument("--limit", type=int, help="Run only the first N collected jobs")
    parser.add_argument("--voice-prompt", default="NATF2.pt")
    parser.add_argument("--voice-prompt-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42_424_242)
    parser.add_argument(
        "--conversation-prompt",
        default=DEFAULT_CONVERSATION_PROMPT,
        help="Prompt for non-interruption tasks",
    )
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--mimi-weight", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--copy-companions",
        action="store_true",
        help="Copy missing input/annotation files into each result sample",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if not args.conversation_prompt.strip():
        raise ValueError("--conversation-prompt must not be empty")
    for option in ("model_weight", "mimi_weight", "tokenizer"):
        value = getattr(args, option)
        if value is not None and not value.is_file():
            raise ValueError(f"--{option.replace('_', '-')} not found: {value}")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("--output must not be the dataset directory or inside it")
    if output.exists() and not output.is_dir():
        raise ValueError(f"--output is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty; pass --resume or choose another")


def main() -> None:
    args = parse_args()
    try:
        _validate_args(args)
        output_root = args.output.resolve()
        jobs = collect_fdb_jobs(
            args.dataset.resolve(),
            output_root,
            set(args.splits) if args.splits else None,
            args.conversation_prompt,
        )
        if args.limit is not None:
            jobs = jobs[: args.limit]
        if args.copy_companions:
            copied = sum(copy_companion_files(job) for job in jobs)
            print(f"[System] Copied {copied} missing companion file(s)")
        manifest_path = output_root / MANIFEST_NAME
        manifest = load_manifest(manifest_path) if args.resume else {}
    except ValueError as exc:
        raise SystemExit(f"Input error: {exc}") from exc

    pending = [job for job in jobs if not job_is_complete(job, manifest)]
    print(
        f"[System] {len(jobs)} selected job(s); "
        f"{len(jobs) - len(pending)} complete; {len(pending)} pending"
    )
    if not pending:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    try:
        from personaplex_backend import (
            BackendConfig,
            PersonaPlexBackend,
            resolve_voice_prompt,
        )

        voice_prompt = resolve_voice_prompt(
            args.voice_prompt, args.voice_prompt_dir, args.hf_repo
        )
        backend = PersonaPlexBackend(
            BackendConfig(
                hf_repo=args.hf_repo,
                moshi_weight=args.model_weight,
                mimi_weight=args.mimi_weight,
                tokenizer=args.tokenizer,
                device=args.device,
                seed=args.seed,
                cpu_offload=args.cpu_offload,
                greedy=False,
                temp_audio=0.8,
                temp_text=0.7,
                topk_audio=250,
                topk_text=25,
            ),
            voice_prompt,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"Backend error: {exc}") from exc

    started = time.perf_counter()
    for index, job in enumerate(pending, start=1):
        item_started = time.perf_counter()
        backend.generate(
            job.input_wav,
            job.prompt,
            job.output_wav,
            job.output_tokens,
            seed=args.seed,
        )
        if not (
            job.output_wav.is_file()
            and job.output_wav.stat().st_size > 0
            and _valid_token_dump(job.output_tokens)
        ):
            raise RuntimeError(f"Incomplete generated outputs for {job.key}")
        record_completion(manifest_path, manifest, _job_signature(job))
        elapsed = time.perf_counter() - started
        rate = elapsed / index
        eta = (len(pending) - index) * rate
        print(
            f"[{index}/{len(pending)}] {job.key}: "
            f"{time.perf_counter() - item_started:.1f}s, ETA {eta / 60:.1f}m"
        )

    print(f"[System] Done in {(time.perf_counter() - started) / 60:.1f}m")


if __name__ == "__main__":
    main()
