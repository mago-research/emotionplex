# EmotionPlex

4×4 emotion duplex speech for LoRA-finetuning [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1).

Qwen writes a user line in one emotion and four agent replies (happy, sad, angry, neutral). Text-to-speech renders them. Each sample is 24 kHz stereo: **left = agent**, **right = user**, plus a JSON sidecar (`alignments`, `text_prompt`) and `train.jsonl` / `val.jsonl`.

## Dataset

```bash
cd dataset
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

**Two fixed voices** — one agent, one user, shared across the set. Demo voices are downloaded on first run.

```bash
python generate_dataset_dual_spk.py --out-dir outputs
```

**Many voices** — each dialogue picks an agent/user pair from a clip directory. The user channel gets a room-tone bed and a short random hush. Sad agent turns use a fixed emotion-tag prefix. Point `--voice-dir` at a folder whose `manifest.json` looks like:

```json
{"speakers": [{"id": "spk01", "wav": "clips/spk01.wav", "text": "...", "gender": "F"}]}
```

```bash
python generate_dataset_multi_spk.py \
  --voice-dir voice_prompts \
  --user-roomtone /path/to/roomtone.wav \
  --out-dir outputs
```

`--voice-source bootstrap` synthesizes two reference voices instead. `--samples-per-mapping` is the number of users per emotion (default 90). `--skip-tts` writes the dialogues only.

## Train

`finetune/configs/emotion_v4.yaml` is the multi-speaker run: LoRA rank 512, system prompt on, depformer LoRA trained, 8000 steps. Edit the jsonl paths if your outputs live elsewhere. `gen_eval` stays off; that eval stack is not in this repo.

```bash
cd finetune
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
torchrun --nproc-per-node 1 -m train configs/emotion_v4.yaml
```

Training code is adapted from [tin-computer/personaplex-finetune](https://github.com/tin-computer/personaplex-finetune) (MIT) and [Kyutai moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) (Apache-2.0), and loads LoRA from [moshi](https://github.com/kyutai-labs/moshi). PersonaPlex weights stay on Hugging Face under the NVIDIA model license.

## License

The source code in this repository is under the Apache License, Version 2.0 (`LICENSE`). Boson weights, Boson source, and any modified Boson architecture stay outside this tree and are fetched when a dataset script runs.

Running either dataset script downloads [multimodalart/higgs-audio-v3-tts-4b-transformers](https://huggingface.co/multimodalart/higgs-audio-v3-tts-4b-transformers). That download is the Boson checkpoint packaged for Transformers, plus remote code. Those artifacts are Higgs Materials and stay under the Boson research and non-commercial license. Using them requires accepting that license on Hugging Face. Commercial use, including production, a hosted API, embedding the model in a product, or resale, needs a separate written license from Boson AI USA, Inc.

Copies of the agreements are in this repository:

- [`licenses/BOSON-HIGGS-AUDIO-V3-LICENSE.txt`](licenses/BOSON-HIGGS-AUDIO-V3-LICENSE.txt) — Boson Higgs Audio v3 Research and Non-Commercial License, last updated May 21, 2026. This is the text shipped with the Transformers port the scripts download.
- [`licenses/BOSON-HIGGS-TTS-3-LICENSE.txt`](licenses/BOSON-HIGGS-TTS-3-LICENSE.txt) — Boson Higgs TTS 3 Research and Non-Commercial License, last updated July 8, 2026. This is the current text on the official Boson model repository. Higgs TTS 3 is the same model, also called Higgs Audio v3.

The required attribution, kept verbatim in [`NOTICE`](NOTICE):

> Boson Higgs Audio v3 is licensed under the Boson Higgs Audio v3 Research and Non-Commercial License, Copyright (c) Boson AI USA, Inc. All Rights Reserved.

> Boson Higgs TTS 3 is licensed under the Boson Higgs TTS 3 Research and Non-Commercial License, Copyright (c) Boson AI USA, Inc. All Rights Reserved.

Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights Reserved.

The Transformers port describes the backbone as Qwen3-4B (Apache-2.0, Copyright 2024 Alibaba Cloud). Dialogue text is written by Qwen2.5-3B-Instruct (Apache-2.0).

If a later release includes the Boson weights, a modified Boson architecture, or code derived from Boson, that release cannot use `license: apache-2.0` or `license: mit`. Its Hugging Face card has to name the Boson license and link the full text, for example:

```yaml
license: other
license_name: boson-higgs-audio-v3-research-and-non-commercial-license
license_link: LICENSE
```

The current official model card uses `license_name: boson-higgs-tts-3-research-and-non-commercial-license`.

### Acceptable use

Use of the downloaded Higgs Materials has to follow the agreements above and the [Boson AI Acceptable Use Policy](https://boson.ai/acceptable-use). In particular, the license prohibits:

- Cloning, simulating, or imitating a real person's voice without that person's explicit consent, and using reference audio you do not have the rights to use.
- Impersonation, fraud, harassment, stalking, threats, defamation, or deception.
- Deceptive political persuasion, election deception, campaign impersonation, voter suppression, or synthetic audio that falsely attributes statements to a public figure without authorization and the disclosures the law requires.
- Identifying, verifying, tracking, surveilling, or profiling a person from voice or other biometric characteristics, except with a lawful basis and the explicit consent the law requires.
- Generating audio that depicts a minor in a sexual, exploitative, or otherwise harmful context.
- Using the Higgs Materials or their output to train, fine-tune, distill, or improve a non-Boson generative speech, audio, language, or multimodal model, unless Boson has authorized that use in a separate written agreement.

The last restriction is in both agreements (Section IV(b)(i) and Section III). This repository renders speech with Higgs and can fine-tune PersonaPlex, which is not a Boson model, on that speech. Confirm that use with Boson before you rely on this release.
