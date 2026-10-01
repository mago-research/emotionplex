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
