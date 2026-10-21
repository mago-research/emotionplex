"""Shared emotion2vec model metadata and label helpers."""

MODEL_ID = "iic/emotion2vec_base_finetuned"
MODEL_REVISION = "v2.0.4"

MODEL_LABELS_RAW = [
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
]


def label_en(raw_label: str) -> str:
    """Return the English portion of an emotion2vec label."""
    if "/" in raw_label:
        return raw_label.split("/")[-1].strip().lower()
    return raw_label.strip("<>").lower()


MODEL_LABELS_EN = [label_en(label) for label in MODEL_LABELS_RAW]
