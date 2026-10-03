#!/usr/bin/env python3
"""Generate a 4x4 emotion duplex set with a pool of speaker voices.

Same text pipeline as generate_dataset_dual_spk.py, with three audio differences:
each dialogue draws its own agent/user pair from --voice-dir, the user channel
sits on a room-tone bed with a short random hush, and sad agent turns use a
fixed emotion-tag prefix (see TTS_TAG_PREFIX).

  uv venv --python 3.11 && source .venv/bin/activate
  uv pip install -r requirements.txt
  python generate_dataset_multi_spk.py --samples-per-mapping 1 \\
    --voice-source bootstrap --user-roomtone /path/to/roomtone.wav \\
    --curr-emotions happy --res-emotions happy,sad --out-dir outputs

Regen one agent emotion without resynthesizing the user:
  python generate_dataset_multi_spk.py --regen-res sad --out-dir outputs
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

# Shared clusters often inject broken ~/.local packages onto sys.path.
sys.path = [p for p in sys.path if ".local/lib/python" not in str(p).replace("\\", "/")]

try:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}\n"
        "Use the project venv (not conda base / ~/.local):\n"
        "  uv venv --python 3.11 && source .venv/bin/activate\n"
        "  uv pip install -r requirements.txt\n"
        "  which python   # should end with .../dataset/.venv/bin/python"
    ) from exc

# Bail if an outside install still leaked in.
for _mod in (torch, sys.modules.get("transformers")):
    if _mod is None:
        continue
    _origin = getattr(_mod, "__file__", "") or ""
    if ".local/lib/python" in _origin.replace("\\", "/"):
        raise SystemExit(
            f"Using packages from ~/.local ({_origin}), not your venv.\n"
            "  deactivate; source .venv/bin/activate\n"
            "  uv pip install -r requirements.txt\n"
            "  python -c \"import transformers; print(transformers.__file__)\""
        )

SAMPLE_RATE = 24_000
GAP_SEC = 0.2
PAD_PRE_MAX_SEC = 1.0
PAD_POST_MAX_SEC = 1.0
ROOMTONE_DBFS = -50.0
ROOMTONE_CROSSFADE_SEC = 0.015
ROOMTONE_BANG_RATIO = 4.0
DEFAULT_VOICE_DIR = Path(__file__).resolve().parent / "voice_prompts" / "libritts_r"
DEFAULT_ROOMTONE = Path(__file__).resolve().parent / "assets" / "roomtone" / "classroom.wav"

TTS_REPO = "multimodalart/higgs-audio-v3-tts-4b-transformers"
LLM_REPO = "Qwen/Qwen2.5-3B-Instruct"

EMOTIONS = ("happy", "sad", "angry", "neutral")

# 4-class labels → TTS emotion nouns sampled at wrap time.
TTS_EMOTION_CHOICES = {
    "happy": ("amusement", "enthusiasm"),
    "sad": ("sadness",),
    "angry": ("anger",),
    "neutral": (),
}
# Full prefix used instead of sampling TTS_EMOTION_CHOICES when set.
TTS_TAG_PREFIX = {
    "sad": (
        "<|emotion:sadness|>"
        "<|emotion:longing|>"
        "<|prosody:pitch_low|>"
        "<|prosody:speed_slow|>"
        "<|prosody:expressive_high|>"
    ),
}

# Optional expressive SFX. Value is (sfx_name, required_onomatopoeia) —
# tag and sound must be adjacent.
# laughter/crying: only between two sentences, 40%. screaming: prepended, 10%.
SFX_PROB = 0.10
SFX_PROB_MID_TURN = 0.40
SFX_CHOICES = {
    "happy": (("laughter", "Hahahaha"),),
    "sad": (("crying", "Aaahhh"), ("crying", "Ooohhh")),
    "angry": (("screaming", "Ahh"),),
}
_MID_TURN_SFX = frozenset({"laughter", "crying"})

# Sized for ~360 users/emotion (4x v2): prefer unique topic per index.
TOPIC_BANK = (
    "work",
    "family",
    "weekend plans",
    "good news",
    "a complaint",
    "travel",
    "health",
    "hobbies",
    "school",
    "neighbors",
    "food",
    "money",
    "a friend",
    "the weather",
    "a mistake",
    "a celebration",
    "a job interview",
    "moving apartments",
    "a delayed flight",
    "traffic",
    "a broken phone",
    "cooking dinner",
    "a birthday party",
    "dating",
    "pets",
    "exercise",
    "sleep",
    "shopping",
    "a package delivery",
    "group project",
    "a landlord issue",
    "childcare",
    "a wedding",
    "retirement plans",
    "a concert",
    "sports",
    "gaming",
    "a streaming show",
    "social media",
    "a reunion",
    "home repairs",
    "gardening",
    "a medical appointment",
    "car trouble",
    "public transit",
    "a noisy roommate",
    "taxes",
    "a raise or bonus",
    "being late",
    "lost keys",
    "a surprise visit",
    "vacation photos",
    "learning a language",
    "volunteering",
    "a restaurant booking",
    "coffee plans",
    "an apology",
    "a favor",
    "bad customer service",
    "a power outage",
    "wifi problems",
    "a new hobby",
    "decluttering",
    "a book recommendation",
    "music practice",
    "an exam result",
    "internship news",
    "a roommate search",
    "holiday travel",
    "a family dinner",
    "an awkward meeting",
    "a forgotten password",
    "laundry",
    "grocery shopping",
    "a haircut",
    "new shoes",
    "a canceled plan",
    "waiting in line",
    "a noisy neighbor",
    "a spilled coffee",
    "finding a parking spot",
    # --- expanded for 4x scale ---
    "a promotion",
    "a deadline",
    "working from home",
    "a team meeting",
    "a new coworker",
    "quitting a job",
    "a performance review",
    "overtime",
    "a work trip",
    "email overload",
    "a client call",
    "office politics",
    "a presentation",
    "a missed meeting",
    "changing careers",
    "a side hustle",
    "student loans",
    "rent going up",
    "buying a house",
    "selling a car",
    "a broken laptop",
    "a dead battery",
    "a flat tire",
    "getting a ticket",
    "missing the bus",
    "a subway delay",
    "rideshare surge pricing",
    "airport security",
    "lost luggage",
    "a hotel booking",
    "a road trip",
    "camping",
    "a beach day",
    "hiking",
    "a picnic",
    "a rainy weekend",
    "a heat wave",
    "snow day",
    "allergy season",
    "a dentist visit",
    "getting glasses",
    "a prescription refill",
    "physical therapy",
    "a gym membership",
    "starting to run",
    "yoga class",
    "a sports injury",
    "meal prep",
    "trying a new recipe",
    "ordering takeout",
    "a food delivery mixup",
    "a burned dinner",
    "baking bread",
    "coffee shop plans",
    "brunch plans",
    "a dinner reservation",
    "a potluck",
    "a bake sale",
    "farmers market",
    "a thrift store find",
    "online shopping",
    "a return policy fight",
    "a sold-out item",
    "a warranty claim",
    "a phone upgrade",
    "a software update",
    "losing phone signal",
    "a video call fail",
    "a group chat drama",
    "an old classmate",
    "a childhood friend",
    "meeting the parents",
    "a first date",
    "a breakup",
    "getting engaged",
    "anniversary plans",
    "a baby shower",
    "a school play",
    "parent-teacher night",
    "homework help",
    "college applications",
    "a scholarship",
    "graduating",
    "a dorm move-in",
    "a study group",
    "all-nighter studying",
    "a pop quiz",
    "group presentation nerves",
    "switching majors",
    "a gap year",
    "learning to drive",
    "a driving test",
    "renewing a license",
    "insurance paperwork",
    "a bank appointment",
    "a bounced payment",
    "splitting a bill",
    "venmo confusion",
    "a subscription charge",
    "budgeting",
    "saving for a trip",
    "a garage sale",
    "donating clothes",
    "assembling furniture",
    "painting a room",
    "a leaky faucet",
    "a clogged sink",
    "pest control",
    "a broken heater",
    "air conditioning failing",
    "a smoke alarm beeping",
    "locking yourself out",
    "a spare key hunt",
    "a plant dying",
    "adopting a dog",
    "a vet visit",
    "a pet sitter",
    "dog training",
    "a cat at the door",
    "birdwatching",
    "aquarium care",
    "a board game night",
    "a movie night",
    "a podcast recommendation",
    "a museum visit",
    "an art show",
    "a comedy night",
    "karaoke",
    "a dance class",
    "learning an instrument",
    "a open mic night",
    "a sports game",
    "fantasy league",
    "a marathon",
    "swimming lessons",
    "climbing gym",
    "bike commuting",
    "a flat bike tire",
    "a parking ticket",
    "street cleaning day",
    "moving day",
    "packing boxes",
    "hiring movers",
    "a storage unit",
    "downsizing",
    "in-laws visiting",
    "hosting overnight guests",
    "a family argument",
    "inheritance paperwork",
    "planning a funeral",
    "a hospital visit",
    "taking care of a parent",
    "a sibling rivalry",
    "a cousin's wedding",
    "holiday gift shopping",
    "decorating for the holidays",
    "new year resolutions",
    "a surprise party",
    "an office birthday cake",
    "a farewell party",
    "a networking event",
    "a conference",
    "public speaking",
    "a podcast interview",
    "writing a blog post",
    "starting a newsletter",
    "a photography hobby",
    "editing photos",
    "a 3d printer project",
    "building a pc",
    "a console release",
    "a game update",
    "losing a ranked match",
    "a streaming marathon",
    "a book club",
    "finishing a novel",
    "a library fine",
    "a magazine subscription",
    "learning to cook",
    "cutting caffeine",
    "trying meditation",
    "a sleep study",
    "insomnia",
    "a nightmare",
    "jet lag",
    "daylight saving time",
    "a calendar mixup",
    "double-booking",
    "rescheduling plans",
    "ghosting a group chat",
    "an unread message pile",
    "a wrong number text",
    "a spam call",
    "identity theft scare",
    "a phishing email",
    "two-factor authentication",
    "resetting a router",
    "smart home glitches",
    "a vacuum robot stuck",
    "a dishwasher leak",
    "replacing a lightbulb",
    "hanging a picture",
    "a DIY fail",
    "borrowing tools",
    "returning a borrowed item late",
    "a library book overdue",
    "losing a wallet",
    "finding a lost item",
    "a coat check mixup",
    "umbrella left behind",
    "sunglasses broken",
    "a torn jacket",
    "laundry shrinkage",
    "a dry cleaning delay",
    "shoe repair",
    "tailoring clothes",
    "a fashion sale",
    "trying on clothes",
    "a hair dye mistake",
    "a bad haircut",
    "nails appointment",
    "a spa day",
    "massage booking",
    "skincare routine",
    "allergy test results",
    "blood test results",
    "a vaccine appointment",
    "eye exam",
    "hearing test",
    "physical checkup",
    "mental health day",
    "therapy appointment",
    "a support group",
    "quitting smoking",
    "cutting back on sugar",
    "a diet challenge",
    "intermittent fasting",
    "meal delivery box",
    "a restaurant review",
    "a cooking competition show",
    "trying spicy food",
    "a food allergy scare",
    "ordering the wrong dish",
    "splitting dessert",
    "a coffee spill on notes",
    "a laptop left at cafe",
    "studying in a cafe",
    "coworking space",
    "a quiet office day",
    "open office noise",
    "hot-desking chaos",
    "badge not working",
    "printer jam",
    "running out of sticky notes",
    "whiteboard markers dry",
    "a brainstorming session",
    "pitching an idea",
    "getting feedback",
    "a rejected proposal",
    "a successful launch",
    "hitting a sales target",
    "losing a client",
    "onboarding someone",
    "mentoring an intern",
    "asking for time off",
    "a sick day",
    "covering a shift",
    "shift swap",
    "night shift",
    "commute construction",
    "bike lane closed",
    "ferry delay",
    "train strike",
    "uber cancelation",
    "taxi overcharge",
    "toll road surprise",
    "ev charger queue",
    "running out of gas",
    "car inspection",
    "new license plates",
    "a scratch on the car",
    "scratching someone's car",
    "parallel parking fail",
    "valet mixup",
    "garage elevator broken",
    "apartment package theft",
    "mailroom closed",
    "amazon locker full",
    "wrong address delivery",
    "a return label issue",
    "customs fee surprise",
    "passport renewal",
    "visa paperwork",
    "a travel advisory",
    "currency exchange",
    "language barrier abroad",
    "getting lost in a new city",
    "a map app fail",
    "sightseeing plans",
    "museum tickets sold out",
    "theme park lines",
    "a roller coaster",
    "a zoo visit",
    "aquarium tickets",
    "a botanical garden",
    "planting tomatoes",
    "composting",
    "mowing the lawn",
    "leaf cleanup",
    "shoveling snow",
    "ice on the steps",
    "a frozen pipe",
    "water bill spike",
    "electric bill spike",
    "negotiating a bill",
    "switching internet providers",
    "cable outage",
    "streaming password share",
    "a show getting canceled",
    "a season finale",
    "spoiler alert fail",
    "a concert ticket scam",
    "nosebleed seats",
    "meeting the band",
    "a sports autograph",
    "tailgating",
    "playoff tickets",
    "a hometown team win",
    "a tough loss",
    "coaching kids soccer",
    "a school fundraiser",
    "bake sale leftovers",
    "raffle tickets",
    "charity run",
    "blood donation",
    "food bank volunteering",
    "tutoring someone",
    "babysitting",
    "pet sitting overnight",
    "house sitting",
    "watering plants for a neighbor",
    "borrowing sugar",
    "a package left with a neighbor",
    "a building fire drill",
    "elevator stuck briefly",
    "stairwell workout",
    "a rooftop hangout",
    "balcony plants",
    "noise complaint letter",
    "quiet hours reminder",
    "a party next door",
    "early morning construction",
    "street festival",
    "farmers market crowd",
    "a food truck",
    "a pop-up shop",
    "vintage market",
    "record store find",
    "comic book shop",
    "board game cafe",
    "escape room",
    "trivia night",
    "bingo night",
    "bowling night",
    "mini golf",
    "arcade night",
    "laser tag",
    "ice skating",
    "roller skating",
    "a pottery class",
    "a painting class",
    "a cooking class",
    "a wine tasting",
    "a brewery tour",
    "a farmers cooking demo",
)

# Hints so cross-emotion replies stay content-coherent with user valence.
MAPPING_HINTS: dict[tuple[str, str], str] = {
    ("happy", "happy"): "Match their excitement; celebrate with them.",
    ("happy", "sad"): (
        "Keep their news as good news, but reply bittersweet about your own angle "
        "(e.g. you'll miss them). Never apologize as if their news were bad."
    ),
    ("happy", "angry"): (
        "Stay clear their news is good; reply with heat about a related unfairness — not at them."
    ),
    ("happy", "neutral"): "Stay grounded; react practically or ask a light follow-up.",
    ("sad", "happy"): "Stay warm and encouraging without forcing cheerfulness onto their pain.",
    ("sad", "sad"): "Meet them gently; shared heaviness is enough.",
    ("sad", "angry"): "Reply with indignation on their behalf; don't congratulate the hardship.",
    ("sad", "neutral"): "Stay calm and open; a short check-in or question is fine.",
    ("angry", "happy"): "Offer a constructive bright side without dismissing the unfairness.",
    ("angry", "sad"): "Reply with quiet concern; shared weight, not pep talk.",
    ("angry", "angry"): "Share the heat about the unfairness; don't attack them.",
    ("angry", "neutral"): "Stay even; ask what happened or what they want to do next.",
    ("neutral", "happy"): "Brighten the reply with genuine enthusiasm about their point.",
    ("neutral", "sad"): "Reply subdued and caring; no forced cheer.",
    ("neutral", "angry"): "Push back firmly about unfairness in what they raised.",
    ("neutral", "neutral"): "Reply plain and natural.",
}

# Lexical cues for rejecting valence-incoherent agent replies.
_USER_POSITIVE_CUES = (
    "great news",
    "good news",
    "promotion",
    "promoted",
    "won",
    "congratulations",
    "can't wait",
    "cannot wait",
    "so excited",
    "amazing",
    "awesome",
    "i got",
    "we got",
    "got accepted",
    "engaged",
    "married",
    "birthday",
    "celebrate",
)
_USER_NEGATIVE_CUES = (
    "struggling",
    "hurts",
    "unfair",
    "lost",
    "fired",
    "broke up",
    "passed away",
    "dying",
    "hate",
    "awful",
    "terrible",
    "can't believe they",
    "so upset",
    "devastated",
)
_AGENT_CONDOLENCE_CUES = (
    "sorry to hear",
    "i'm sorry to hear",
    "im sorry to hear",
    "that's terrible",
    "thats terrible",
    "that's awful",
    "thats awful",
    "that's so sad",
    "condolences",
    "my condolences",
)
_AGENT_CONGRATS_CUES = (
    "congratulations",
    "congrats",
    "so happy for you",
    "thrilled for you",
    "that's amazing",
    "thats amazing",
    "that's wonderful",
    "thats wonderful",
)

SPEAKER_AGENT = "SPEAKER_BROKER"
SPEAKER_USER = "SPEAKER_CLIENT"
# PersonaPlex system prompt; `{emotion}` is filled with the agent res_emotion.
DEFAULT_TEXT_PROMPT = "Answer with a {emotion} tone."

TAG_RE = re.compile(r"<\|[^|]+\|>")
QUOTE_WRAP_RE = re.compile(r'^["“\'`]+|["”\'`]+$')


@dataclass
class UserUtterance:
    user_id: str
    curr_emotion: str
    topic: str
    text: str


@dataclass
class DuplexSample:
    id: str
    user_id: str
    curr_emotion: str
    res_emotion: str
    user_text: str
    agent_text: str
    topic: str = ""


@dataclass
class VoiceRef:
    audio: torch.Tensor  # mono float [L]
    sample_rate: int
    text: str | None
    path: Path | None = None
    name: str = ""
    gender: str = ""


def parse_emotion_list(raw: str) -> list[str]:
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in EMOTIONS]
    if bad:
        raise SystemExit(f"Unknown emotions {bad}; choose from {list(EMOTIONS)}")
    if not items:
        raise SystemExit("Emotion list is empty")
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--samples-per-mapping",
        type=int,
        default=90,
        help="Users per curr emotion (= duplex samples per curr→res mapping)",
    )
    p.add_argument(
        "--curr-emotions",
        type=str,
        default=",".join(EMOTIONS),
        help="Comma-separated user/curr emotions",
    )
    p.add_argument(
        "--res-emotions",
        type=str,
        default=",".join(EMOTIONS),
        help="Comma-separated agent/res emotions",
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--llm-repo", type=str, default=LLM_REPO)
    p.add_argument("--tts-repo", type=str, default=TTS_REPO)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu / mps (default: auto)")
    p.add_argument(
        "--voice0",
        type=Path,
        default=None,
        help="Reference WAV for agent / SPEAKER_BROKER (left)",
    )
    p.add_argument(
        "--voice1",
        type=Path,
        default=None,
        help="Reference WAV for user / SPEAKER_CLIENT (right)",
    )
    p.add_argument("--voice0-text", type=str, default=None, help="Transcript of voice0")
    p.add_argument("--voice1-text", type=str, default=None, help="Transcript of voice1")
    p.add_argument(
        "--voice-dir",
        type=Path,
        default=DEFAULT_VOICE_DIR,
        help=(
            "Speaker-clip directory with manifest.json "
            f"(default: {DEFAULT_VOICE_DIR}). "
            'speakers: [{id, wav, text, gender}, ...].'
        ),
    )
    p.add_argument(
        "--voice-source",
        type=str,
        choices=("auto", "bootstrap", "files", "dir"),
        default="auto",
        help=(
            "Where speaker refs come from. "
            "auto: --voice0/--voice1 if given, else --voice-dir; "
            "dir: load clips from --voice-dir/manifest.json; "
            "bootstrap: synthesize two reference voices; "
            "files: require --voice0/--voice1."
        ),
    )
    p.add_argument(
        "--bootstrap-voices",
        action="store_true",
        help="Alias for --voice-source bootstrap.",
    )
    p.add_argument("--gap-sec", type=float, default=GAP_SEC)
    p.add_argument(
        "--user-roomtone",
        type=Path,
        default=DEFAULT_ROOMTONE,
        help="24 kHz mono room-tone WAV used as the user-channel bed",
    )
    p.add_argument(
        "--pad-pre-max-sec",
        type=float,
        default=PAD_PRE_MAX_SEC,
        help="Sample prepend hush uniformly in [0, this] seconds",
    )
    p.add_argument(
        "--pad-post-max-sec",
        type=float,
        default=PAD_POST_MAX_SEC,
        help="Sample append hush uniformly in [0, this] seconds",
    )
    p.add_argument(
        "--user-roomtone-dbfs",
        type=float,
        default=ROOMTONE_DBFS,
        help="Target RMS of the user-channel roomtone bed in dBFS",
    )
    p.add_argument("--temperature", type=float, default=0.8, help="TTS sampling temperature")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--best-of-two",
        action="store_true",
        help="Synthesize two TTS candidates per turn and keep the higher score",
    )
    p.add_argument("--keep-candidates", action="store_true", help="Save both TTS candidates")
    p.add_argument("--skip-tts", action="store_true", help="Only write dialogue JSONL (no audio)")
    p.add_argument(
        "--dialogues-jsonl",
        type=Path,
        default=None,
        help="Reuse existing dialogues.jsonl; skip LLM",
    )
    p.add_argument(
        "--users-jsonl",
        type=Path,
        default=None,
        help="Optional users.jsonl to load with --dialogues-jsonl",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip duplex outputs that already exist; reuse cached user WAVs",
    )
    p.add_argument(
        "--regen-res",
        type=str,
        default=None,
        help=(
            "Comma-separated agent res_emotions to force-regenerate "
            "(agent TTS + stereo remux only). Reuses dialogues.jsonl and "
            "user WAVs; other mappings are left untouched. "
            "Implies loading --out-dir/dialogues.jsonl unless --dialogues-jsonl is set."
        ),
    )
    p.add_argument(
        "--text-prompt",
        type=str,
        default=DEFAULT_TEXT_PROMPT,
        help=(
            "PersonaPlex text_prompt template for sidecars. "
            "Use {emotion} for the agent res_emotion "
            '(default: "Answer with a {emotion} tone.").'
        ),
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of each emotion mapping held out for val (default 0.05). "
            "Uses the same sample indices across all mappings for balance and "
            "to avoid shared-user leakage. Set 0 to disable."
        ),
    )
    p.add_argument(
        "--split-only",
        action="store_true",
        help=(
            "Only (re)write train.jsonl / val.jsonl from existing stereo_wav + "
            "dialogues.jsonl; skip LLM and TTS."
        ),
    )
    p.add_argument(
        "--patch-text-prompts",
        action="store_true",
        help=(
            "Rewrite text_prompt in existing stereo_wav/*.json from res_emotion "
            "(no LLM/TTS). Uses --text-prompt template."
        ),
    )
    return p.parse_args()


def pick_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free_memory(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def strip_tts_tags(text: str) -> str:
    return TAG_RE.sub("", text).strip()


def clean_plain_utterance(raw: str) -> str:
    """Normalize LLM output into a short plain spoken line."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Prefer first non-empty line; drop role prefixes.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(user|assistant|agent|a|b)\s*:\s*", "", line, flags=re.I)
        line = QUOTE_WRAP_RE.sub("", line).strip()
        line = strip_tts_tags(line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            return line
    return ""


def validate_utterance(text: str, *, emotion: str, max_chars: int = 280) -> str | None:
    """Return cleaned text or None if unusable."""
    text = clean_plain_utterance(text)
    if len(text) < 8:
        return None
    if len(text) > max_chars:
        return None
    # Reject meta / instruction leakage.
    low = text.lower()
    banned = (
        "as an ai",
        "here is",
        "here's a",
        "json",
        "<|",
        "emotion:",
        "i will write",
    )
    if any(b in low for b in banned):
        return None
    # Prefer not naming the target emotion explicitly.
    if re.search(rf"\b{re.escape(emotion)}\b", low):
        # Soft reject only when the emotion word is the clear subject.
        if low.startswith(emotion) or f"i am {emotion}" in low or f"i'm {emotion}" in low:
            return None
    return text


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(c in text for c in cues)


def reply_mismatches_user_valence(user_text: str, agent_text: str) -> bool:
    """True when the reply treats good news as bad (or vice versa)."""
    user_l = user_text.lower()
    agent_l = agent_text.lower()
    user_pos = _has_any(user_l, _USER_POSITIVE_CUES)
    user_neg = _has_any(user_l, _USER_NEGATIVE_CUES)
    if user_pos and not user_neg and _has_any(agent_l, _AGENT_CONDOLENCE_CUES):
        return True
    if user_neg and not user_pos and _has_any(agent_l, _AGENT_CONGRATS_CUES):
        return True
    return False


def validate_agent_reply(
    text: str,
    *,
    user_text: str,
    res_emotion: str,
    max_chars: int = 280,
) -> str | None:
    text = validate_utterance(text, emotion=res_emotion, max_chars=max_chars)
    if text is None:
        return None
    if reply_mismatches_user_valence(user_text, text):
        return None
    return text


def _stable_index(seed_key: str, n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16) % n


def _stable_chance(seed_key: str, p: float) -> bool:
    # 1000 buckets → about p probability.
    bucket = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16) % 1000
    return bucket < int(round(p * 1000))


def split_spoken_sentences(text: str) -> list[str]:
    """Split on . ! ? while keeping the terminator on each sentence."""
    parts = re.findall(r".+?(?:[.!?]+[\"']*|\Z)", text.strip(), flags=re.S)
    return [p.strip() for p in parts if p.strip()]


def wrap_emotion_tag(emotion: str, text: str, *, seed_key: str) -> str:
    """Prepend a sampled emotion tag (+ optional SFX) for TTS.

    Neutral has no emotion nouns, so the spoken line is left untagged.
    Laughter/crying only fire on two-or-more-sentence turns, inserted after the
    first sentence. Screaming is still prepended.
    """
    plain = strip_tts_tags(text).strip()
    if emotion in TTS_TAG_PREFIX:
        prefix = TTS_TAG_PREFIX[emotion]
    else:
        choices = TTS_EMOTION_CHOICES.get(emotion, ())
        prefix = ""
        if choices:
            tag = choices[_stable_index(f"{seed_key}:emo", len(choices))]
            prefix = f"<|emotion:{tag}|>"

    sfx_opts = SFX_CHOICES.get(emotion)
    if not sfx_opts:
        return f"{prefix}{plain}"

    sfx, sound = sfx_opts[_stable_index(f"{seed_key}:sfx_pick", len(sfx_opts))]
    sentences = split_spoken_sentences(plain)
    if sfx in _MID_TURN_SFX and len(sentences) < 2:
        return f"{prefix}{plain}"
    sfx_p = SFX_PROB_MID_TURN if sfx in _MID_TURN_SFX else SFX_PROB
    if not _stable_chance(f"{seed_key}:sfx", sfx_p):
        return f"{prefix}{plain}"

    # Onomatopoeia goes immediately after the sfx tag, with no space.
    sfx_chunk = f"<|sfx:{sfx}|>{sound}"
    if sfx in _MID_TURN_SFX:
        first, rest = sentences[0], " ".join(sentences[1:])
        if rest[:1].isalpha():
            return f"{prefix}{first}{sfx_chunk} {rest}"
        return f"{prefix}{first}{sfx_chunk}{rest}"

    prefix += sfx_chunk
    if plain[:1].isalpha():
        return f"{prefix}, {plain}"
    return f"{prefix}{plain}"


def tokenize_words(text: str) -> list[str]:
    cleaned = strip_tts_tags(text)
    words = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9'\s]", cleaned)
    return [w for w in words if re.search(r"[A-Za-z0-9]", w)]


def proportional_word_timings(
    words: list[str], start: float, end: float
) -> list[tuple[str, float, float]]:
    """Spread words across [start, end] proportional to character length."""
    if not words:
        return []
    weights = [max(len(w), 1) for w in words]
    total = float(sum(weights))
    span = max(end - start, 1e-3)
    out: list[tuple[str, float, float]] = []
    t = start
    for w, weight in zip(words, weights):
        dur = span * (weight / total)
        out.append((w, t, t + dur))
        t += dur
    if out:
        w, s, _ = out[-1]
        out[-1] = (w, s, end)
    return out


def user_id_for(curr: str, index: int) -> str:
    return f"user_{curr}_{index:03d}"


def duplex_id_for(curr: str, res: str, index: int) -> str:
    return f"{curr}_{res}_{index:03d}"


def format_text_prompt(template: str, res_emotion: str) -> str:
    """Fill `{emotion}` with the agent response emotion for PersonaPlex."""
    try:
        return template.format(emotion=res_emotion)
    except (KeyError, ValueError):
        # Allow literal prompts without placeholders.
        return template


def topic_for(curr: str, index: int) -> str:
    # Stable rotation across topics, offset by emotion for variety.
    offset = EMOTIONS.index(curr) if curr in EMOTIONS else 0
    return TOPIC_BANK[(index + offset) % len(TOPIC_BANK)]


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------


def _resample_mono(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    x = wav.detach().float().cpu().numpy()
    n_out = max(int(round(len(x) * float(target_sr) / float(orig_sr))), 1)
    t_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    y = np.interp(t_new, t_old, x).astype(np.float32)
    return torch.from_numpy(y)


def load_mono(path: Path, target_sr: int = SAMPLE_RATE) -> tuple[torch.Tensor, int]:
    pcm, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(pcm.mean(axis=1))
    if sr != target_sr:
        wav = _resample_mono(wav, sr, target_sr)
        sr = target_sr
    return wav.float().cpu(), sr


def write_wav(path: Path, wav: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2:
        out = arr.T  # soundfile expects [T, C]
    else:
        raise ValueError(f"Expected 1D or 2D audio, got shape {arr.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), out, sr)


def frame_energies(wav: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    frame = max(int(sr * frame_ms / 1000.0), 1)
    n = len(wav) // frame
    if n == 0:
        return np.array([0.0], dtype=np.float64)
    clipped = wav[: n * frame].reshape(n, frame)
    return np.sqrt(np.mean(clipped**2, axis=1) + 1e-12)


def score_tts_candidate(wav: np.ndarray, text: str, sr: int) -> float:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size < int(0.15 * sr):
        return -1e6

    peak = float(np.max(np.abs(wav)))
    if peak < 1e-4:
        return -1e6

    x = wav / max(peak, 1e-8)
    clip_frac = float(np.mean(np.abs(wav) > 0.99))
    if clip_frac > 0.02:
        return -1e6

    spoken = strip_tts_tags(text)
    n_chars = max(len(re.sub(r"\s+", " ", spoken)), 1)
    dur = wav.size / float(sr)
    cps = n_chars / max(dur, 1e-3)
    rate_score = -abs(cps - 12.5)
    if cps < 4.0 or cps > 28.0:
        rate_score -= 15.0

    rms = float(np.sqrt(np.mean(wav**2) + 1e-12))
    db = 20.0 * np.log10(rms + 1e-12)
    level_score = -abs(db - (-18.0))

    energies = frame_energies(x, sr)
    silence_frac = float(np.mean(energies < 0.02))
    if silence_frac > 0.65:
        return -1e6

    log_e = np.log(energies + 1e-8)
    express = float(np.std(log_e))
    if energies.size > 2:
        express += 0.5 * float(np.std(np.diff(log_e)))

    headroom = -abs(peak - 0.7)
    return (
        2.5 * rate_score
        + 1.2 * level_score
        + 4.0 * express
        - 8.0 * silence_frac
        - 40.0 * clip_frac
        + 0.5 * headroom
    )


def choose_best_candidate(
    candidates: list[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, float, int]:
    best_i = int(np.argmax([s for _, s in candidates]))
    wav, score = candidates[best_i]
    return wav, score, best_i


# ---------------------------------------------------------------------------
# Dialogue generation (Qwen) — C+D plain text
# ---------------------------------------------------------------------------


USER_SYSTEM_PROMPT = """You write one everyday spoken line for a person in a casual conversation.

Hard rules:
- English only. Exactly 1–2 short sentences.
- Speak naturally; act the requested emotion through wording and tone.
- Never name or announce the emotion (do not say happy/sad/angry/neutral).
- No quotes, no JSON, no markdown, no role labels, no stage directions.
- Output ONLY the spoken line."""


AGENT_SYSTEM_PROMPT = """You reply as a conversational partner to one user message.

Hard rules:
- English only. Exactly 1–2 short sentences.
- First understand the CONTENT of what the user said (good news vs bad news vs neutral).
- Your reply MUST answer that specific message (not a parallel monologue).
- Keep content coherent with what they said (e.g. do not congratulate hard news,
  do not treat good news as bad). Explicit acknowledgment is optional — use it only
  when it fits the context and your reply emotion.
- Then speak in the requested reply emotion. Emotion is your tone/mood, not a license
  to misread their news (e.g. sad reply to a promotion can be bittersweet about missing
  them — never "sorry to hear that" about the promotion itself).
- Never name or announce the emotion (do not say happy/sad/angry/neutral).
- No quotes, no JSON, no markdown, no role labels, no stage directions.
- Output ONLY the spoken reply."""


def apply_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def llm_generate_text(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    seed: int,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 128,
) -> str:
    prompt = apply_chat_prompt(tokenizer, messages)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    new_tokens = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_user_line(
    model,
    tokenizer,
    *,
    curr_emotion: str,
    topic: str,
    seed: int,
) -> str:
    user_msg = (
        f"Write one spoken user line.\n"
        f"emotion={curr_emotion}\n"
        f"topic={topic}\n"
        f"variety_seed={seed}"
    )
    messages = [
        {"role": "system", "content": USER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = llm_generate_text(model, tokenizer, messages, seed=seed)
    text = validate_utterance(raw, emotion=curr_emotion)
    if text:
        return text
    # One retry cooler.
    raw = llm_generate_text(
        model,
        tokenizer,
        messages,
        seed=seed + 997,
        temperature=0.7,
        top_p=0.9,
    )
    text = validate_utterance(raw, emotion=curr_emotion)
    if text:
        return text
    raise ValueError(f"user line failed validation: {raw[:160]!r}")


def generate_agent_reply(
    model,
    tokenizer,
    *,
    user_text: str,
    curr_emotion: str,
    res_emotion: str,
    seed: int,
) -> str:
    hint = MAPPING_HINTS.get(
        (curr_emotion, res_emotion),
        "Reply in the requested emotion without misreading the valence of their message.",
    )
    user_msg = (
        f"User said: {user_text}\n"
        f"User emotion (for context only): {curr_emotion}\n"
        f"Your reply emotion: {res_emotion}\n"
        f"How to reply: {hint}\n"
        f"Remember: do not misread the valence of their message."
    )
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = llm_generate_text(model, tokenizer, messages, seed=seed)
    text = validate_agent_reply(raw, user_text=user_text, res_emotion=res_emotion)
    if text:
        return text
    # Retry with an explicit anti-pattern reminder.
    retry_messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                user_msg
                + "\nBad example to avoid: saying 'sorry to hear that' after good news, "
                "or 'congratulations' after bad news."
            ),
        },
    ]
    raw = llm_generate_text(
        model,
        tokenizer,
        retry_messages,
        seed=seed + 991,
        temperature=0.7,
        top_p=0.9,
    )
    text = validate_agent_reply(raw, user_text=user_text, res_emotion=res_emotion)
    if text:
        return text
    raise ValueError(f"agent reply failed validation: {raw[:160]!r}")


def generate_dataset_texts(
    llm_repo: str,
    device: torch.device,
    *,
    curr_emotions: list[str],
    res_emotions: list[str],
    samples_per_mapping: int,
    seed: int,
) -> tuple[list[UserUtterance], list[DuplexSample]]:
    print(f"Loading LLM {llm_repo} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_repo)
    model = AutoModelForCausalLM.from_pretrained(
        llm_repo,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    users: list[UserUtterance] = []
    duplexes: list[DuplexSample] = []

    for curr in curr_emotions:
        for i in range(samples_per_mapping):
            uid = user_id_for(curr, i)
            topic = topic_for(curr, i)
            print(f"\n=== user {uid} (topic={topic}) ===")
            user_seed = seed + (int(hashlib.md5(uid.encode()).hexdigest()[:8], 16) % 100_000)
            try:
                user_text = generate_user_line(
                    model,
                    tokenizer,
                    curr_emotion=curr,
                    topic=topic,
                    seed=user_seed,
                )
            except Exception as exc:
                print(f"  [warn] {uid} failed: {exc}; using fallback")
                user_text = _fallback_user_line(curr, topic)

            users.append(
                UserUtterance(
                    user_id=uid,
                    curr_emotion=curr,
                    topic=topic,
                    text=user_text,
                )
            )
            print(f"  user: {user_text}")

            for res in res_emotions:
                did = duplex_id_for(curr, res, i)
                agent_seed = seed + (
                    int(hashlib.md5(did.encode()).hexdigest()[:8], 16) % 100_000
                )
                try:
                    agent_text = generate_agent_reply(
                        model,
                        tokenizer,
                        user_text=user_text,
                        curr_emotion=curr,
                        res_emotion=res,
                        seed=agent_seed,
                    )
                except Exception as exc:
                    print(f"  [warn] {did} agent failed: {exc}; using fallback")
                    agent_text = _fallback_agent_reply(curr, res, user_text)

                duplexes.append(
                    DuplexSample(
                        id=did,
                        user_id=uid,
                        curr_emotion=curr,
                        res_emotion=res,
                        user_text=user_text,
                        agent_text=agent_text,
                        topic=topic,
                    )
                )
                print(f"  {res}: {agent_text}")

    free_memory(model, tokenizer)
    return users, duplexes


def _fallback_user_line(emotion: str, topic: str) -> str:
    templates = {
        "happy": f"I just got great news about {topic}, I can't stop smiling.",
        "sad": f"I've been struggling with {topic} lately, and it really hurts.",
        "angry": f"I can't believe what happened with {topic}, it's so unfair.",
        "neutral": f"I wanted to catch you up on {topic} when you have a minute.",
    }
    return templates.get(emotion, templates["neutral"])


def _fallback_agent_reply(curr_emotion: str, res_emotion: str, _user_text: str) -> str:
    """Deterministic coherent fallbacks when Qwen fails validation."""
    templates = {
        ("happy", "happy"): "That's amazing, I'm so happy for you!",
        ("happy", "sad"): (
            "I'm really happy for you, though it makes me a little sad knowing "
            "things will change."
        ),
        ("happy", "angry"): (
            "I'm glad you got that, though it still frustrates me how long they "
            "made you wait."
        ),
        ("happy", "neutral"): "That's good news, what happens next for you?",
        ("sad", "happy"): (
            "I'm sorry it's been hard, but I believe things can get better from here."
        ),
        ("sad", "sad"): "That sounds really heavy, I'm here with you.",
        ("sad", "angry"): (
            "That's not okay, it makes me angry they put you through that."
        ),
        ("sad", "neutral"): "Thanks for telling me, do you want to talk about it?",
        ("angry", "happy"): (
            "I get why you're upset, and I still think you've got a way through this."
        ),
        ("angry", "sad"): "I hear how unfair that feels, and it honestly makes me sad too.",
        ("angry", "angry"): "You're right to be mad, that shouldn't have happened.",
        ("angry", "neutral"): "Okay, walk me through what happened.",
        ("neutral", "happy"): "Nice, that actually makes me pretty excited for you.",
        ("neutral", "sad"): "I hear you, and somehow that leaves me feeling a bit low.",
        ("neutral", "angry"): "That detail bothers me, it doesn't sound fair at all.",
        ("neutral", "neutral"): "Got it, tell me a bit more when you can.",
    }
    return templates.get(
        (curr_emotion, res_emotion),
        "Thanks for sharing that, tell me a bit more when you're ready.",
    )


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_users_jsonl(path: Path) -> list[UserUtterance]:
    users: list[UserUtterance] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        users.append(
            UserUtterance(
                user_id=str(item["user_id"]),
                curr_emotion=str(item["curr_emotion"]),
                topic=str(item.get("topic", "")),
                text=clean_plain_utterance(str(item["text"])),
            )
        )
    return users


def load_dialogues_jsonl(path: Path) -> list[DuplexSample]:
    duplexes: list[DuplexSample] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        duplexes.append(
            DuplexSample(
                id=str(item["id"]),
                user_id=str(item["user_id"]),
                curr_emotion=str(item["curr_emotion"]),
                res_emotion=str(item["res_emotion"]),
                user_text=clean_plain_utterance(str(item["user_text"])),
                agent_text=clean_plain_utterance(str(item["agent_text"])),
                topic=str(item.get("topic", "")),
            )
        )
    return duplexes


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


def load_tts(tts_repo: str, device: torch.device):
    print(f"Loading TTS {tts_repo} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(tts_repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        tts_repo,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    if hasattr(model, "get_audio_codec"):
        model.get_audio_codec()
    return model, tokenizer


def synthesize_once(
    model,
    tokenizer,
    text: str,
    voice: VoiceRef | None,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p if top_p < 1.0 else None,
        "top_k": top_k if top_k > 0 else None,
    }
    if voice is not None:
        kwargs["reference_audio"] = voice.audio
        kwargs["reference_sample_rate"] = voice.sample_rate
        if voice.text:
            kwargs["reference_text"] = voice.text

    with torch.no_grad():
        wav = model.generate_speech(text, tokenizer, **kwargs)
    if wav is None or wav.numel() == 0:
        return np.zeros(0, dtype=np.float32)
    return wav.detach().float().cpu().numpy().reshape(-1)


def synthesize_turn(
    model,
    tokenizer,
    text: str,
    voice: VoiceRef | None,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    base_seed: int,
    best_of_two: bool,
    keep_candidates: bool,
    cand_dir: Path | None,
    cand_prefix: str,
) -> tuple[np.ndarray, float, int]:
    if not best_of_two:
        wav = synthesize_once(
            model,
            tokenizer,
            text,
            voice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            seed=base_seed,
        )
        score = score_tts_candidate(wav, text, SAMPLE_RATE)
        print(f"      score={score:.3f} dur={len(wav)/SAMPLE_RATE:.2f}s")
        return wav, score, 0

    cands: list[tuple[np.ndarray, float]] = []
    for k in range(2):
        wav = synthesize_once(
            model,
            tokenizer,
            text,
            voice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            seed=base_seed + k * 17,
        )
        score = score_tts_candidate(wav, text, SAMPLE_RATE)
        cands.append((wav, score))
        print(f"      candidate {k}: score={score:.3f} dur={len(wav)/SAMPLE_RATE:.2f}s")
        if keep_candidates and cand_dir is not None:
            cpath = cand_dir / f"{cand_prefix}_c{k}.wav"
            if wav.size:
                write_wav(cpath, wav, SAMPLE_RATE)
    best_wav, best_score, best_i = choose_best_candidate(cands)
    return best_wav, best_score, best_i


def bootstrap_voices(model, tokenizer, out_dir: Path, seed: int) -> tuple[VoiceRef, VoiceRef]:
    voices_dir = out_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    scripts = [
        (
            0,
            "<|prosody:pitch_low|><|prosody:expressive_low|>"
            "Hello, my name is Alex. I speak clearly and calmly in everyday conversation.",
            "Hello, my name is Alex. I speak clearly and calmly in everyday conversation.",
        ),
        (
            1,
            "<|prosody:pitch_high|><|prosody:expressive_low|>"
            "Hello, my name is Jordan. I speak clearly and calmly in everyday conversation.",
            "Hello, my name is Jordan. I speak clearly and calmly in everyday conversation.",
        ),
    ]
    refs: list[VoiceRef] = []
    for spk, tagged, plain in scripts:
        wav = synthesize_once(
            model,
            tokenizer,
            tagged,
            voice=None,
            temperature=0.4,
            top_p=0.9,
            top_k=50,
            max_new_tokens=1024,
            seed=seed + 1000 + spk,
        )
        path = voices_dir / f"speaker{spk}_ref.wav"
        write_wav(path, wav, SAMPLE_RATE)
        audio, sr = load_mono(path, SAMPLE_RATE)
        refs.append(
            VoiceRef(
                audio=audio,
                sample_rate=sr,
                text=plain,
                path=path,
                name=f"speaker{spk}",
            )
        )
        print(f"  bootstrapped voice {spk} -> {path}")
    return refs[0], refs[1]


def load_voice_files(args: argparse.Namespace) -> tuple[VoiceRef, VoiceRef]:
    if not args.voice0 or not args.voice1:
        raise SystemExit("--voice-source files requires both --voice0 and --voice1")
    a0, sr0 = load_mono(args.voice0, SAMPLE_RATE)
    a1, sr1 = load_mono(args.voice1, SAMPLE_RATE)
    return (
        VoiceRef(
            audio=a0,
            sample_rate=sr0,
            text=args.voice0_text,
            path=args.voice0,
            name="voice0",
        ),
        VoiceRef(
            audio=a1,
            sample_rate=sr1,
            text=args.voice1_text,
            path=args.voice1,
            name="voice1",
        ),
    )


def load_voice_dir(voice_dir: Path) -> list[VoiceRef]:
    """Load speaker clips listed in voice_dir/manifest.json."""
    manifest_path = voice_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No {manifest_path}\n"
            'Expected {"speakers": [{"id", "wav", "text", "gender"}, ...]}\n'
            "or pass --voice-source bootstrap / --voice0 and --voice1."
        )
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("speakers") or []
    if len(rows) < 2:
        raise SystemExit(f"{manifest_path} has fewer than 2 speakers")
    refs: list[VoiceRef] = []
    for row in rows:
        wav_path = Path(str(row["wav"]))
        if not wav_path.is_absolute():
            wav_path = (voice_dir / wav_path).resolve()
        audio, sr = load_mono(wav_path, SAMPLE_RATE)
        text = str(row.get("text") or "").strip() or None
        txt_sidecar = wav_path.with_suffix(".txt")
        if text is None and txt_sidecar.exists():
            text = txt_sidecar.read_text().strip() or None
        refs.append(
            VoiceRef(
                audio=audio,
                sample_rate=sr,
                text=text,
                path=wav_path,
                name=str(row.get("id") or wav_path.stem),
                gender=str(row.get("gender") or "").upper(),
            )
        )
        print(
            f"  voice {refs[-1].name} {refs[-1].gender or '?'} "
            f"-> {wav_path} ({audio.numel() / sr:.1f}s)"
        )
    return refs


def pick_voice_pair(voices: list[VoiceRef], user_id: str) -> tuple[VoiceRef, VoiceRef]:
    """Return (agent_voice, user_voice). Opposite gender when the pool allows."""
    if len(voices) < 2:
        raise SystemExit("Need at least two speaker refs (agent + user)")
    if len(voices) == 2:
        return voices[0], voices[1]

    females = [v for v in voices if v.gender == "F"]
    males = [v for v in voices if v.gender == "M"]
    user_female = bool(_stable_index(f"{user_id}:user_gender", 2))
    if females and males:
        user_pool = females if user_female else males
        agent_pool = males if user_female else females
    else:
        user_pool = agent_pool = voices
    user_v = user_pool[_stable_index(f"{user_id}:user_voice", len(user_pool))]
    agent_v = agent_pool[_stable_index(f"{user_id}:agent_voice", len(agent_pool))]
    if agent_v.name == user_v.name:
        alt = [v for v in voices if v.name != user_v.name]
        agent_v = alt[_stable_index(f"{user_id}:agent_voice_alt", len(alt))]
    return agent_v, user_v


def resolve_voices(args: argparse.Namespace, model, tokenizer) -> list[VoiceRef]:
    """Return the speaker-ref pool (2 files/demos, or LibriTTS-R clips)."""
    source = args.voice_source
    if args.bootstrap_voices:
        source = "bootstrap"
    voice_dir = args.voice_dir or DEFAULT_VOICE_DIR
    if source == "auto":
        if args.voice0 and args.voice1:
            source = "files"
        else:
            source = "dir"

    if source == "files":
        return list(load_voice_files(args))
    if source == "dir":
        print(f"Loading LibriTTS-R voice clips from {voice_dir}...")
        return load_voice_dir(voice_dir)
    if source == "bootstrap":
        print("Bootstrapping two reference voices...")
        return list(bootstrap_voices(model, tokenizer, args.out_dir, args.seed))
    raise SystemExit(f"Unknown --voice-source {source!r}")


# ---------------------------------------------------------------------------
# Stereo packing (StyleTalk / PersonaPlex)
# ---------------------------------------------------------------------------


def _rng_for_sample(seed: int, sample_id: str) -> random.Random:
    digest = hashlib.sha256(f"{int(seed)}:{sample_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def _wrap_slice(src: np.ndarray, start: int, n: int) -> np.ndarray:
    src = np.asarray(src, dtype=np.float32).reshape(-1)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    length = len(src)
    if length == 0:
        return np.zeros(n, dtype=np.float32)
    start = int(start) % length
    if start + n <= length:
        return src[start : start + n].copy()
    idx = (np.arange(n, dtype=np.int64) + start) % length
    return src[idx]


def _scale_to_dbfs(wav: np.ndarray, target_dbfs: float) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return wav
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-12:
        return wav
    target = float(10.0 ** (float(target_dbfs) / 20.0))
    return (wav * (target / rms)).astype(np.float32)


def sample_roomtone_crop(
    source: np.ndarray,
    n: int,
    rng: random.Random,
    sr: int,
    *,
    bang_ratio: float = ROOMTONE_BANG_RATIO,
    n_tries: int = 24,
) -> np.ndarray:
    """Contiguous wrap-crop of length n; retry windows much louder than the source median."""
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if source.size == 0:
        return np.zeros(n, dtype=np.float32)
    src_e = frame_energies(source, sr)
    median_rms = float(np.median(src_e))
    threshold = max(median_rms * float(bang_ratio), 1e-8)
    crop = _wrap_slice(source, 0, n)
    length = len(source)
    for _ in range(max(int(n_tries), 1)):
        start = rng.randrange(length) if length > 1 else 0
        candidate = _wrap_slice(source, start, n)
        peak = float(np.max(frame_energies(candidate, sr)))
        crop = candidate
        if peak <= threshold:
            break
    return crop


def _fade_edges(wav: np.ndarray, fade_n: int) -> np.ndarray:
    """Equal-power fade in/out on a speech take so it sits on a continuous bed."""
    out = np.asarray(wav, dtype=np.float32).reshape(-1).copy()
    fade_n = min(int(fade_n), len(out) // 2)
    if fade_n < 1 or out.size == 0:
        return out
    t = np.linspace(0.0, 1.0, fade_n, endpoint=True, dtype=np.float32)
    w_in = np.sin(0.5 * np.pi * t)
    w_out = np.cos(0.5 * np.pi * t)
    out[:fade_n] *= w_in
    out[-fade_n:] *= w_out
    return out


def compose_duplex_stereo(
    user_wav: np.ndarray,
    agent_wav: np.ndarray,
    gap_sec: float,
    sr: int = SAMPLE_RATE,
    *,
    roomtone: np.ndarray,
    rng: random.Random,
    pad_pre_max_sec: float = PAD_PRE_MAX_SEC,
    pad_post_max_sec: float = PAD_POST_MAX_SEC,
    roomtone_dbfs: float = ROOMTONE_DBFS,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """Timeline [pre][user][gap][agent][post]; left=agent, right=user.

    Right channel is a contiguous roomtone crop for the whole clip, with user
    speech mixed on top. Left is zeros except the agent take.
    Returns stereo [2, T], user_dur, agent_start, agent_dur, pad_pre_sec, pad_post_sec.
    """
    user = np.asarray(user_wav, dtype=np.float32).reshape(-1)
    agent = np.asarray(agent_wav, dtype=np.float32).reshape(-1)
    gap_n = int(round(gap_sec * sr))
    pad_pre_n = int(round(rng.uniform(0.0, max(float(pad_pre_max_sec), 0.0)) * sr))
    pad_post_n = int(round(rng.uniform(0.0, max(float(pad_post_max_sec), 0.0)) * sr))
    total_n = pad_pre_n + len(user) + gap_n + len(agent) + pad_post_n

    bed = sample_roomtone_crop(roomtone, total_n, rng, sr)
    bed = _scale_to_dbfs(bed, roomtone_dbfs)

    left = np.zeros(total_n, dtype=np.float32)
    right = bed
    user_start = pad_pre_n
    agent_start_n = pad_pre_n + len(user) + gap_n
    fade_n = int(round(ROOMTONE_CROSSFADE_SEC * sr))
    if len(user):
        right[user_start : user_start + len(user)] += _fade_edges(user, fade_n)
    left[agent_start_n : agent_start_n + len(agent)] = agent

    user_dur = len(user) / float(sr)
    pad_pre_sec = pad_pre_n / float(sr)
    pad_post_sec = pad_post_n / float(sr)
    agent_start = agent_start_n / float(sr)
    agent_dur = len(agent) / float(sr)
    stereo = np.vstack((left, right))
    return stereo, user_dur, agent_start, agent_dur, pad_pre_sec, pad_post_sec


def build_sidecar(
    sample: DuplexSample,
    *,
    wav_path: Path,
    user_path: Path,
    agent_path: Path,
    user_dur: float,
    agent_start: float,
    agent_dur: float,
    gap_sec: float,
    pad_pre_sec: float,
    pad_post_sec: float,
    text_prompt: str,
    agent_voice: VoiceRef,
    user_voice: VoiceRef,
    roomtone_path: Path | None = None,
    roomtone_dbfs: float | None = None,
) -> dict:
    user_start = float(pad_pre_sec)
    user_end = user_start + float(user_dur)
    agent_end = float(agent_start) + float(agent_dur)
    user_words = tokenize_words(sample.user_text)
    agent_words = tokenize_words(sample.agent_text)
    alignments: list[list] = []
    for word, start, end in proportional_word_timings(user_words, user_start, user_end):
        alignments.append([word, [float(start), float(end)], SPEAKER_USER])
    for word, start, end in proportional_word_timings(agent_words, agent_start, agent_end):
        alignments.append([word, [float(start), float(end)], SPEAKER_AGENT])

    turns = [
        {
            "index": 0,
            "speaker": SPEAKER_USER,
            "start": float(user_start),
            "end": float(user_end),
            "text": sample.user_text,
            "role": "user",
            "path": str(user_path),
        },
        {
            "index": 1,
            "speaker": SPEAKER_AGENT,
            "start": float(agent_start),
            "end": float(agent_end),
            "text": sample.agent_text,
            "role": "agent",
            "path": str(agent_path),
        },
    ]
    return {
        "id": sample.id,
        "user_id": sample.user_id,
        "alignments": alignments,
        "turns": turns,
        "text_prompt": text_prompt,
        "styletalk": {
            "curr_emotion": sample.curr_emotion,
            "res_emotion": sample.res_emotion,
            "topic": sample.topic,
            "user_id": sample.user_id,
        },
        "sample_rate": SAMPLE_RATE,
        "gap_sec": gap_sec,
        "pad_pre_sec": float(pad_pre_sec),
        "pad_post_sec": float(pad_post_sec),
        "user_roomtone": str(roomtone_path) if roomtone_path is not None else None,
        "user_roomtone_dbfs": float(roomtone_dbfs) if roomtone_dbfs is not None else None,
        "aligner": "proportional",
        "channel_layout": {"left": SPEAKER_AGENT, "right": SPEAKER_USER},
        "voice_prompt": str(agent_voice.path) if agent_voice.path else None,
        "voices": {
            "agent": {
                "name": agent_voice.name or None,
                "gender": agent_voice.gender or None,
                "path": str(agent_voice.path) if agent_voice.path else None,
                "text": agent_voice.text,
            },
            "user": {
                "name": user_voice.name or None,
                "gender": user_voice.gender or None,
                "path": str(user_voice.path) if user_voice.path else None,
                "text": user_voice.text,
            },
        },
        "tagged_text": {
            "user": wrap_emotion_tag(
                sample.curr_emotion,
                sample.user_text,
                seed_key=f"{sample.user_id}:user",
            ),
            "agent": wrap_emotion_tag(
                sample.res_emotion,
                sample.agent_text,
                seed_key=f"{sample.id}:agent",
            ),
        },
        "wav": str(wav_path),
        "duration": float(agent_end + pad_post_sec),
    }


def ensure_user_wav(
    model,
    tokenizer,
    user: UserUtterance,
    user_voice: VoiceRef,
    turns_dir: Path,
    args: argparse.Namespace,
    cand_dir: Path | None,
) -> tuple[np.ndarray, Path]:
    path = turns_dir / f"{user.user_id}.wav"
    if args.resume and path.exists() and path.stat().st_size > 0:
        audio, _ = load_mono(path, SAMPLE_RATE)
        print(f"  reuse user wav {path}")
        return audio.numpy(), path

    tagged = wrap_emotion_tag(
        user.curr_emotion, user.text, seed_key=f"{user.user_id}:user"
    )
    print(f"  TTS user {user.user_id}: {tagged}")
    seed = args.seed + (int(hashlib.md5(user.user_id.encode()).hexdigest()[:8], 16) % 10_000)
    wav, score, _ = synthesize_turn(
        model,
        tokenizer,
        tagged,
        user_voice,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        base_seed=seed,
        best_of_two=args.best_of_two,
        keep_candidates=args.keep_candidates,
        cand_dir=cand_dir,
        cand_prefix=user.user_id,
    )
    if score <= -1e5 or wav.size == 0:
        print("    [warn] weak user TTS; keeping anyway")
    write_wav(path, wav, SAMPLE_RATE)
    return wav, path


def synthesize_agent_wav(
    model,
    tokenizer,
    sample: DuplexSample,
    agent_voice: VoiceRef,
    turns_dir: Path,
    args: argparse.Namespace,
    cand_dir: Path | None,
    *,
    force: bool = False,
) -> tuple[np.ndarray, Path]:
    path = turns_dir / f"{sample.id}_agent.wav"
    if not force and args.resume and path.exists() and path.stat().st_size > 0:
        audio, _ = load_mono(path, SAMPLE_RATE)
        print(f"  reuse agent wav {path}")
        return audio.numpy(), path

    tagged = wrap_emotion_tag(
        sample.res_emotion, sample.agent_text, seed_key=f"{sample.id}:agent"
    )
    print(f"  TTS agent {sample.id}: {tagged}")
    seed = args.seed + (int(hashlib.md5(sample.id.encode()).hexdigest()[:8], 16) % 10_000)
    wav, score, _ = synthesize_turn(
        model,
        tokenizer,
        tagged,
        agent_voice,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        base_seed=seed,
        best_of_two=args.best_of_two,
        keep_candidates=args.keep_candidates,
        cand_dir=cand_dir,
        cand_prefix=f"{sample.id}_agent",
    )
    if score <= -1e5 or wav.size == 0:
        print("    [warn] weak agent TTS; keeping anyway")
    write_wav(path, wav, SAMPLE_RATE)
    return wav, path


def duplex_index(sample_id: str) -> int:
    """Parse trailing index from ids like happy_sad_007."""
    return int(str(sample_id).rsplit("_", 1)[-1])


def balanced_emotion_split(
    rows: list[dict],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    """Hold out the same indices across every curr→res mapping.

    Keeps val perfectly balanced over emotion mappings and keeps all res
    variants of a user index together (no shared-user train/val leakage).
    Manifest rows are PersonaPlex-style {"path", "duration"} only.
    """
    if val_fraction <= 0 or not rows:
        slim = [{"path": r["path"], "duration": r["duration"]} for r in rows]
        info = {
            "val_fraction": val_fraction,
            "val_indices": [],
            "train": len(slim),
            "val": 0,
            "per_mapping": {},
        }
        return slim, [], info

    by_mapping: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = f"{r['curr_emotion']}->{r['res_emotion']}"
        by_mapping[key].append(r)

    min_n = min(len(items) for items in by_mapping.values())
    # Same index universe 0..min_n-1 assumed for a complete grid.
    # Half-up so 90 * 0.05 -> 5 per mapping (not banker's 4).
    k = int(math.floor(min_n * val_fraction + 0.5))
    if min_n > 1:
        k = min(max(k, 0), min_n - 1)
    else:
        k = 0  # tiny smoke sets stay fully in train

    indices = list(range(min_n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_indices = set(indices[:k])

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    per_mapping: dict[str, dict] = {}

    for key, items in sorted(by_mapping.items()):
        items = sorted(items, key=lambda r: duplex_index(r["id"]))
        m_train = 0
        m_val = 0
        for r in items:
            slim = {"path": r["path"], "duration": float(r["duration"])}
            idx = duplex_index(r["id"])
            if idx in val_indices:
                val_rows.append(slim)
                m_val += 1
            else:
                train_rows.append(slim)
                m_train += 1
        per_mapping[key] = {"train": m_train, "val": m_val}

    info = {
        "val_fraction": val_fraction,
        "target_val_per_mapping": k,
        "val_indices": sorted(val_indices),
        "seed": seed,
        "train": len(train_rows),
        "val": len(val_rows),
        "per_mapping": per_mapping,
    }
    return train_rows, val_rows, info


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_manifests(
    out_dir: Path,
    all_rows: list[dict],
    mapping_stats: dict[str, dict],
    *,
    val_fraction: float,
    seed: int,
) -> dict:
    train_rows, val_rows, split_info = balanced_emotion_split(
        all_rows, val_fraction=val_fraction, seed=seed
    )
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    (out_dir / "manifest_by_mapping.json").write_text(
        json.dumps(mapping_stats, indent=2) + "\n"
    )
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_info, indent=2) + "\n"
    )
    return split_info


def _res_emotion_from_sidecar(meta: dict, path: Path) -> str | None:
    style = meta.get("styletalk") or {}
    res = style.get("res_emotion") or meta.get("res_emotion")
    if res:
        return str(res).strip().lower()
    # Fallback: ids like happy_sad_007 → res = sad
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[1] in EMOTIONS:
        return parts[1]
    return None


def patch_text_prompts(out_dir: Path, template: str) -> tuple[int, int]:
    """Update text_prompt in existing sidecars. Returns (updated, skipped)."""
    stereo_dir = out_dir / "stereo_wav"
    if not stereo_dir.is_dir():
        raise SystemExit(f"No stereo_wav directory at {stereo_dir}")

    updated = 0
    skipped = 0
    for meta_path in sorted(stereo_dir.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            print(f"  [warn] skip {meta_path.name}: {exc}")
            skipped += 1
            continue
        res = _res_emotion_from_sidecar(meta, meta_path)
        if res not in EMOTIONS:
            print(f"  [warn] skip {meta_path.name}: unknown res_emotion={res!r}")
            skipped += 1
            continue
        new_prompt = format_text_prompt(template, res)
        if meta.get("text_prompt") == new_prompt:
            skipped += 1
            continue
        meta["text_prompt"] = new_prompt
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        updated += 1
    return updated, skipped


def collect_rows_from_outputs(out_dir: Path, duplexes: list[DuplexSample]) -> tuple[list[dict], dict]:
    """Build manifest rows from existing stereo sidecars."""
    stereo_dir = out_dir / "stereo_wav"
    rows: list[dict] = []
    mapping_stats: dict[str, dict] = {}
    for sample in duplexes:
        wav_path = stereo_dir / f"{sample.id}.wav"
        meta_path = stereo_dir / f"{sample.id}.json"
        if not wav_path.exists() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            dur = float(meta.get("duration", 0.0))
        except Exception:
            dur = 0.0
        mapping_key = f"{sample.curr_emotion}->{sample.res_emotion}"
        mapping_stats.setdefault(
            mapping_key,
            {"count": 0, "total_duration_sec": 0.0, "ids": []},
        )
        mapping_stats[mapping_key]["count"] += 1
        mapping_stats[mapping_key]["total_duration_sec"] += dur
        mapping_stats[mapping_key]["ids"].append(sample.id)
        rows.append(
            {
                "id": sample.id,
                "curr_emotion": sample.curr_emotion,
                "res_emotion": sample.res_emotion,
                "path": str(wav_path.resolve()),
                "duration": dur,
            }
        )
    return rows, mapping_stats


def main() -> None:
    args = parse_args()
    if args.samples_per_mapping < 1:
        raise SystemExit("--samples-per-mapping must be >= 1")
    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in [0, 1)")

    if args.patch_text_prompts:
        updated, skipped = patch_text_prompts(args.out_dir, args.text_prompt)
        print(
            f"Patched text_prompt in {args.out_dir / 'stereo_wav'}: "
            f"updated={updated} unchanged_or_skipped={skipped}"
        )
        return

    regen_res = parse_emotion_list(args.regen_res) if args.regen_res else []
    if regen_res:
        if args.dialogues_jsonl is None:
            args.dialogues_jsonl = args.out_dir / "dialogues.jsonl"
        if not args.dialogues_jsonl.exists():
            raise SystemExit(
                f"--regen-res needs dialogues at {args.dialogues_jsonl}"
            )
        args.resume = True

    curr_emotions = parse_emotion_list(args.curr_emotions)
    res_emotions = parse_emotion_list(args.res_emotions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stereo_dir = args.out_dir / "stereo_wav"
    turns_dir = args.out_dir / "turns"
    cand_dir = args.out_dir / "candidates"
    stereo_dir.mkdir(parents=True, exist_ok=True)
    turns_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_candidates:
        cand_dir.mkdir(parents=True, exist_ok=True)
    else:
        cand_dir = None

    users_path = args.users_jsonl or (args.out_dir / "users.jsonl")
    dialogues_path = args.dialogues_jsonl or (args.out_dir / "dialogues.jsonl")

    if args.split_only:
        if not dialogues_path.exists():
            raise SystemExit(f"--split-only needs dialogues at {dialogues_path}")
        duplexes = load_dialogues_jsonl(dialogues_path)
        rows, mapping_stats = collect_rows_from_outputs(args.out_dir, duplexes)
        if not rows:
            raise SystemExit(f"No stereo_wav sidecars found under {args.out_dir}")
        split_info = write_manifests(
            args.out_dir,
            rows,
            mapping_stats,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print(
            f"Split {len(rows)} samples -> train={split_info['train']} "
            f"val={split_info['val']} (indices={split_info['val_indices']})"
        )
        for key, stats in sorted(split_info["per_mapping"].items()):
            print(f"  {key}: train={stats['train']} val={stats['val']}")
        return

    device = pick_device(args.device)

    if args.dialogues_jsonl:
        duplexes = load_dialogues_jsonl(args.dialogues_jsonl)
        if args.users_jsonl and args.users_jsonl.exists():
            users = load_users_jsonl(args.users_jsonl)
        elif users_path.exists():
            users = load_users_jsonl(users_path)
        else:
            # Reconstruct minimal user records from duplexes.
            seen: dict[str, UserUtterance] = {}
            for d in duplexes:
                if d.user_id not in seen:
                    seen[d.user_id] = UserUtterance(
                        user_id=d.user_id,
                        curr_emotion=d.curr_emotion,
                        topic=d.topic,
                        text=d.user_text,
                    )
            users = list(seen.values())
        print(f"Loaded {len(users)} users, {len(duplexes)} duplexes from JSONL")
        if regen_res:
            n_all = len(duplexes)
            duplexes = [d for d in duplexes if d.res_emotion in regen_res]
            print(
                f"Regen res={','.join(regen_res)}: {len(duplexes)}/{n_all} duplexes"
            )
            if not duplexes:
                raise SystemExit(
                    f"No dialogues with res_emotion in {regen_res}"
                )
    else:
        users, duplexes = generate_dataset_texts(
            llm_repo=args.llm_repo,
            device=device,
            curr_emotions=curr_emotions,
            res_emotions=res_emotions,
            samples_per_mapping=args.samples_per_mapping,
            seed=args.seed,
        )
        save_jsonl(users_path, [asdict(u) for u in users])
        save_jsonl(dialogues_path, [asdict(d) for d in duplexes])
        print(f"Saved users -> {users_path}")
        print(f"Saved dialogues -> {dialogues_path}")

    if args.skip_tts:
        print("--skip-tts set; done.")
        return

    users_by_id = {u.user_id: u for u in users}
    model, tokenizer = load_tts(args.tts_repo, device)
    voices = resolve_voices(args, model, tokenizer)

    user_wav_cache: dict[str, tuple[np.ndarray, Path]] = {}
    all_rows: list[dict] = []
    mapping_stats: dict[str, dict] = {}

    roomtone_path = Path(args.user_roomtone)
    if not roomtone_path.is_file():
        raise SystemExit(
            f"User roomtone WAV not found: {roomtone_path}\n"
            "Pass --user-roomtone with a 24 kHz mono WAV."
        )
    roomtone_tensor, _ = load_mono(roomtone_path, SAMPLE_RATE)
    roomtone_pcm = roomtone_tensor.numpy()
    if roomtone_pcm.size == 0:
        raise SystemExit(f"User roomtone WAV is empty: {roomtone_path}")
    print(
        f"Loaded user roomtone {roomtone_path} "
        f"({len(roomtone_pcm) / SAMPLE_RATE:.1f}s, {args.user_roomtone_dbfs:.1f} dBFS)"
    )

    for sample in duplexes:
        wav_path = stereo_dir / f"{sample.id}.wav"
        meta_path = stereo_dir / f"{sample.id}.json"
        mapping_key = f"{sample.curr_emotion}->{sample.res_emotion}"
        mapping_stats.setdefault(
            mapping_key,
            {"count": 0, "total_duration_sec": 0.0, "ids": []},
        )

        force_regen = bool(regen_res) and sample.res_emotion in regen_res
        if (
            args.resume
            and not force_regen
            and wav_path.exists()
            and meta_path.exists()
        ):
            print(f"\n=== skip existing {sample.id} ===")
            try:
                meta = json.loads(meta_path.read_text())
                dur = float(meta.get("duration", 0.0))
            except Exception:
                dur = 0.0
            all_rows.append(
                {
                    "id": sample.id,
                    "curr_emotion": sample.curr_emotion,
                    "res_emotion": sample.res_emotion,
                    "path": str(wav_path.resolve()),
                    "duration": dur,
                }
            )
            mapping_stats[mapping_key]["count"] += 1
            mapping_stats[mapping_key]["total_duration_sec"] += dur
            mapping_stats[mapping_key]["ids"].append(sample.id)
            continue

        print(f"\n=== {sample.id} ({mapping_key}) ===")
        user = users_by_id.get(sample.user_id)
        if user is None:
            user = UserUtterance(
                user_id=sample.user_id,
                curr_emotion=sample.curr_emotion,
                topic=sample.topic,
                text=sample.user_text,
            )

        agent_voice, user_voice = pick_voice_pair(voices, sample.user_id)
        print(
            f"  voices agent={agent_voice.name}({agent_voice.gender or '?'}) "
            f"user={user_voice.name}({user_voice.gender or '?'})"
        )

        if sample.user_id not in user_wav_cache:
            user_wav_cache[sample.user_id] = ensure_user_wav(
                model,
                tokenizer,
                user,
                user_voice,
                turns_dir,
                args,
                cand_dir,
            )
        user_wav, user_path = user_wav_cache[sample.user_id]
        agent_wav, agent_path = synthesize_agent_wav(
            model,
            tokenizer,
            sample,
            agent_voice,
            turns_dir,
            args,
            cand_dir,
            force=force_regen,
        )

        stereo, user_dur, agent_start, agent_dur, pad_pre_sec, pad_post_sec = (
            compose_duplex_stereo(
                user_wav,
                agent_wav,
                gap_sec=args.gap_sec,
                roomtone=roomtone_pcm,
                rng=_rng_for_sample(args.seed, sample.id),
                pad_pre_max_sec=args.pad_pre_max_sec,
                pad_post_max_sec=args.pad_post_max_sec,
                roomtone_dbfs=args.user_roomtone_dbfs,
            )
        )
        peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
        if peak > 1e-6:
            stereo = (0.95 * stereo / peak).astype(np.float32)

        write_wav(wav_path, stereo, SAMPLE_RATE)
        meta = build_sidecar(
            sample,
            wav_path=wav_path,
            user_path=user_path,
            agent_path=agent_path,
            user_dur=user_dur,
            agent_start=agent_start,
            agent_dur=agent_dur,
            gap_sec=args.gap_sec,
            pad_pre_sec=pad_pre_sec,
            pad_post_sec=pad_post_sec,
            text_prompt=format_text_prompt(args.text_prompt, sample.res_emotion),
            agent_voice=agent_voice,
            user_voice=user_voice,
            roomtone_path=roomtone_path,
            roomtone_dbfs=args.user_roomtone_dbfs,
        )
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        dur = float(meta["duration"])
        all_rows.append(
            {
                "id": sample.id,
                "curr_emotion": sample.curr_emotion,
                "res_emotion": sample.res_emotion,
                "path": str(wav_path.resolve()),
                "duration": dur,
            }
        )
        mapping_stats[mapping_key]["count"] += 1
        mapping_stats[mapping_key]["total_duration_sec"] += dur
        mapping_stats[mapping_key]["ids"].append(sample.id)
        print(f"  saved {wav_path} ({dur:.2f}s)")

    if regen_res:
        full = load_dialogues_jsonl(dialogues_path)
        all_rows, mapping_stats = collect_rows_from_outputs(args.out_dir, full)
        print(
            f"Rebuilt manifests from all {len(all_rows)} stereo clips "
            f"(regenerated res={','.join(regen_res)})"
        )

    split_info = write_manifests(
        args.out_dir,
        all_rows,
        mapping_stats,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"\nDone. {len(all_rows)} duplex samples in {args.out_dir}")
    print(
        f"Split -> train={split_info['train']} val={split_info['val']} "
        f"(val_indices={split_info['val_indices']})"
    )
    for key, stats in sorted(mapping_stats.items()):
        split_m = split_info["per_mapping"].get(key, {})
        print(
            f"  {key}: {stats['count']} samples, "
            f"{stats['total_duration_sec']:.1f}s total "
            f"(train={split_m.get('train', 0)} val={split_m.get('val', 0)})"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
