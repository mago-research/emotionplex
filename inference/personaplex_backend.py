"""Reusable PersonaPlex backend for batched offline generation.

Heavy model libraries are imported only when ``PersonaPlexBackend`` is
constructed. The model, streaming generators, and voice prompt are initialized
once and then reused across calls to :meth:`generate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BackendConfig:
    hf_repo: str
    moshi_weight: Path | None
    mimi_weight: Path | None
    tokenizer: Path | None
    device: str
    seed: int
    cpu_offload: bool
    greedy: bool
    temp_audio: float
    temp_text: float
    topk_audio: int
    topk_text: int


def resolve_voice_prompt(
    voice_prompt: str, voice_prompt_dir: Path | None, hf_repo: str
) -> Path:
    """Resolve an explicit file or a PersonaPlex HF voice-prompt basename."""
    candidate = Path(voice_prompt).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.parent != Path("."):
        raise FileNotFoundError(f"Voice prompt file not found: {candidate}")

    try:
        from moshi.offline import _get_voice_prompt_dir
    except ImportError as exc:
        raise RuntimeError(
            "A NVIDIA/PersonaPlex-compatible moshi package is required. "
            "Install inference/requirements.txt."
        ) from exc

    resolved_dir = Path(
        _get_voice_prompt_dir(
            str(voice_prompt_dir) if voice_prompt_dir is not None else None,
            hf_repo,
        )
    )
    resolved = resolved_dir / candidate.name
    if not resolved.is_file():
        raise FileNotFoundError(f"Voice prompt file not found: {resolved}")
    return resolved.resolve()


class PersonaPlexBackend:
    """Load PersonaPlex once and generate many prompted responses."""

    def __init__(self, config: BackendConfig, voice_prompt_path: Path):
        try:
            import sentencepiece
            import torch
            from huggingface_hub import hf_hub_download
            from moshi.models import LMGen, loaders
            from moshi.models.lm import (
                _iterate_audio as lm_iterate_audio,
                encode_from_sphn as lm_encode_from_sphn,
                load_audio as lm_load_audio,
            )
            from moshi.offline import (
                decode_tokens_to_pcm,
                seed_all,
                warmup,
                wrap_with_system_tags,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PersonaPlex inference dependencies are missing. "
                "Install inference/requirements.txt."
            ) from exc

        self._torch = torch
        self._lm_iterate_audio = lm_iterate_audio
        self._lm_encode_from_sphn = lm_encode_from_sphn
        self._lm_load_audio = lm_load_audio
        self._decode_tokens_to_pcm = decode_tokens_to_pcm
        self._wrap_with_system_tags = wrap_with_system_tags
        self._seed_all = seed_all
        self.device = config.device

        if config.seed != -1:
            seed_all(config.seed)

        # Fetching config preserves the NVIDIA loader's expected repository
        # metadata and fails early for an incompatible repo.
        hf_hub_download(config.hf_repo, "config.json")
        mimi_weight = config.mimi_weight or Path(
            hf_hub_download(config.hf_repo, loaders.MIMI_NAME)
        )
        tokenizer = config.tokenizer or Path(
            hf_hub_download(config.hf_repo, loaders.TEXT_TOKENIZER_NAME)
        )
        moshi_weight = config.moshi_weight or Path(
            hf_hub_download(config.hf_repo, loaders.MOSHI_NAME)
        )

        self.mimi = loaders.get_mimi(str(mimi_weight), self.device)
        self.other_mimi = loaders.get_mimi(str(mimi_weight), self.device)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))
        self.lm = loaders.get_moshi_lm(
            str(moshi_weight),
            device=self.device,
            cpu_offload=config.cpu_offload,
        )
        self.lm.eval()

        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            self.lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=self.device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
            use_sampling=not config.greedy,
            temp=config.temp_audio,
            temp_text=config.temp_text,
            top_k=config.topk_audio,
            top_k_text=config.topk_text,
        )
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        warmup(
            self.mimi,
            self.other_mimi,
            self.lm_gen,
            self.device,
            self.frame_size,
        )

        voice_path = str(voice_prompt_path)
        if voice_path.endswith(".pt"):
            self.lm_gen.load_voice_prompt_embeddings(voice_path)
        else:
            self.lm_gen.load_voice_prompt(voice_path)
        self.sample_rate = int(self.mimi.sample_rate)

    def generate(
        self,
        input_wav: Path,
        text_prompt: str,
        output_wav: Path,
        output_tokens: Path,
        seed: int | None = None,
    ) -> Any:
        """Generate one response, write audio/tokens, and return mono PCM."""
        import json

        import numpy as np
        import sphn

        if seed is not None and seed != -1:
            self._seed_all(seed)

        with self._torch.no_grad():
            self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(
                self._wrap_with_system_tags(text_prompt)
            )
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            self.lm_gen.step_system_prompts(self.mimi)
            self.mimi.reset_streaming()

            user_audio = self._lm_load_audio(str(input_wav), self.sample_rate)
            total_target = user_audio.shape[-1]
            generated_frames: list[Any] = []
            generated_text: list[str] = []

            encoded_chunks = self._lm_encode_from_sphn(
                self.mimi,
                self._lm_iterate_audio(
                    user_audio,
                    sample_interval_size=self.lm_gen._frame_size,
                    pad=True,
                ),
                max_batch=1,
            )
            for user_encoded in encoded_chunks:
                for channel in range(user_encoded.shape[-1]):
                    tokens = self.lm_gen.step(
                        user_encoded[:, :, channel : channel + 1]
                    )
                    if tokens is None:
                        continue
                    generated_frames.append(
                        self._decode_tokens_to_pcm(
                            self.mimi, self.other_mimi, self.lm_gen, tokens
                        )
                    )
                    text_token = tokens[0, 0, 0].item()
                    if text_token not in (0, 3):
                        generated_text.append(
                            self.text_tokenizer.id_to_piece(text_token).replace("▁", " ")
                        )
                    else:
                        generated_text.append(
                            ["EPAD", "BOS", "EOS", "PAD"][text_token]
                        )

        if not generated_frames:
            raise RuntimeError(f"No audio generated for {input_wav}")

        output_pcm = np.concatenate(generated_frames, axis=-1)
        if output_pcm.shape[-1] > total_target:
            output_pcm = output_pcm[:total_target]
        elif output_pcm.shape[-1] < total_target:
            output_pcm = np.concatenate(
                [
                    output_pcm,
                    np.zeros(
                        total_target - output_pcm.shape[-1],
                        dtype=output_pcm.dtype,
                    ),
                ],
                axis=-1,
            )

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_tokens.parent.mkdir(parents=True, exist_ok=True)
        sphn.write_wav(str(output_wav), output_pcm, self.sample_rate)
        output_tokens.write_text(
            json.dumps(generated_text, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output_pcm


# Keep the old class name import-compatible for downstream AEQ callers.
ToneEvaluator = PersonaPlexBackend
