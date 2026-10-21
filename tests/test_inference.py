"""CPU-only unit tests for inference job preparation and resume behavior."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


aeq = load_module(
    "test_run_batch_metric3",
    "inference/aeq_metric3/run_batch_metric3.py",
)
fdb = load_module("test_run_fdb", "inference/fdb/run_fdb.py")


class AEQInferenceTests(unittest.TestCase):
    def test_one_entry_builds_four_emotion_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_dir = root / "wavs"
            wav_dir.mkdir()
            (wav_dir / "anger_0001.wav").write_bytes(b"fake wav")
            dataset = root / "dataset.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "id": "sample_0001",
                            "text": "How are you?",
                            "contexts": {
                                "Context": "A short conversation.",
                                "Angry_explanation": "Acted angrily.",
                                "Angry_response": "A response.",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            jobs = aeq.build_metric3_jobs(
                dataset,
                wav_dir,
                limit=None,
                seed=42_424_242,
                prompt_template=aeq.DEFAULT_PROMPT_TEMPLATE,
            )

        self.assertEqual(4, len(jobs))
        self.assertEqual(
            set(aeq.REQUESTED_EMOTIONS),
            {job["requested_emotion"] for job in jobs},
        )
        self.assertEqual(
            {
                "Answer with a happy tone.",
                "Answer with a sad tone.",
                "Answer with a angry tone.",
                "Answer with a neutral tone.",
            },
            {job["text_prompt"] for job in jobs},
        )

    def test_predictions_schema_and_resume_keys(self) -> None:
        expected = {
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
        }
        self.assertEqual(expected, set(aeq.FIELDNAMES))
        self.assertEqual(len(expected), len(aeq.FIELDNAMES))

        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions.csv"
            with predictions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=aeq.FIELDNAMES)
                writer.writeheader()
                writer.writerow(
                    {
                        **dict.fromkeys(aeq.FIELDNAMES, ""),
                        "utt_id": "anger_0001",
                        "requested_emotion": "happy",
                    }
                )
            self.assertEqual(
                {("anger_0001", "happy")},
                aeq.completed_keys(predictions),
            )


class FDBInferenceTests(unittest.TestCase):
    def _fixture(self, root: Path):
        sample = root / "dataset" / "v1_0" / "synthetic_pause_handling" / "001"
        sample.mkdir(parents=True)
        (sample / "input.wav").write_bytes(b"input")
        (sample / "pause.json").write_text("[]", encoding="utf-8")
        output = root / "output"
        job = fdb.collect_fdb_jobs(root / "dataset", output)[0]
        return sample, output, job

    def test_prompt_mapping(self) -> None:
        self.assertEqual(
            fdb.INTERRUPTION_PROMPT,
            fdb.prompt_for_task("synthetic_user_interruption", "neutral"),
        )
        self.assertEqual(
            fdb.INTERRUPTION_PROMPT,
            fdb.prompt_for_task("USER_INTERRUPTION", "neutral"),
        )
        self.assertEqual("neutral", fdb.prompt_for_task("user_backchannel", "neutral"))

    def test_collection_and_manifest_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, output, job = self._fixture(Path(tmp))
            self.assertEqual(
                "v1_0/synthetic_pause_handling/001",
                job.key,
            )
            self.assertFalse(fdb.job_is_complete(job, {}))

            job.output_dir.mkdir(parents=True)
            job.output_wav.write_bytes(b"output")
            job.output_tokens.write_text('[" hello"]\n', encoding="utf-8")
            # Existing outputs alone must never count as completed.
            self.assertFalse(fdb.job_is_complete(job, {}))

            manifest = {}
            manifest_path = output / fdb.MANIFEST_NAME
            fdb.record_completion(
                manifest_path,
                manifest,
                fdb._job_signature(job),
            )
            loaded = fdb.load_manifest(manifest_path)
            self.assertTrue(fdb.job_is_complete(job, loaded))

            job.output_tokens.write_text("{}", encoding="utf-8")
            self.assertFalse(fdb.job_is_complete(job, loaded))

    def test_companion_copy_is_non_destructive_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample, _, job = self._fixture(Path(tmp))
            (sample / "output.wav").write_bytes(b"dataset output")
            (sample / "output.json").write_text('{"dataset": true}', encoding="utf-8")
            (sample / "output_tokens.json").write_text('["dataset"]', encoding="utf-8")
            job.output_dir.mkdir(parents=True)
            job.output_wav.write_bytes(b"generated output")
            (job.output_dir / "output.json").write_text(
                '{"generated": true}', encoding="utf-8"
            )
            job.output_tokens.write_text('["generated"]', encoding="utf-8")

            copied = fdb.copy_companion_files(job)
            self.assertEqual(2, copied)
            self.assertEqual(b"input", (job.output_dir / "input.wav").read_bytes())
            self.assertEqual(
                "[]",
                (job.output_dir / "pause.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(b"generated output", job.output_wav.read_bytes())
            self.assertEqual(
                '{"generated": true}',
                (job.output_dir / "output.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                '["generated"]',
                job.output_tokens.read_text(encoding="utf-8"),
            )
            self.assertEqual(0, fdb.copy_companion_files(job))
            self.assertEqual(b"input", (sample / "input.wav").read_bytes())


if __name__ == "__main__":
    unittest.main()
