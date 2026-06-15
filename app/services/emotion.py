import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any
from app.config import HF_MODEL_NAME

EMOTION_META = {
    0: {"name": "sadness", "tone": "Warm, validating, gentle — never minimize"},
    1: {"name": "joy", "tone": "Encouraging, celebratory, engaging"},
    2: {"name": "love", "tone": "Warm, affirming, relationship-focused"},
    3: {"name": "anger", "tone": "Calm, empathetic, non-confrontational"},
    4: {"name": "fear", "tone": "Reassuring, grounding, structured"},
    5: {"name": "surprise", "tone": "Curious, engaged, open"},
}
LABEL_TO_NAME = {k: v["name"] for k, v in EMOTION_META.items()}

ISOLATION_PHRASES = [
    "nobody understands",
    "no one understands",
    "nobody cares",
    "no one cares",
    "all alone",
    "completely alone",
    "nobody listens",
    "no one listens",
]


class EmotionClassifier:
    """Lazy loader and wrapper for the HuggingFace Sequence Classification model for Emotion Detection."""

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
        self._device = "cpu"

    def _load(self):
        if not self._is_loaded:
            # Force CPU-only to optimize resources in container
            self._tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                HF_MODEL_NAME
            ).to(self._device)
            self._model.eval()
            self._is_loaded = True

    def classify(self, text: str, threshold: float = 0.40) -> Dict[str, Any]:
        # Rule-based fast-track for severe isolation phrases
        text_lower = text.lower().strip()
        if any(phrase in text_lower for phrase in ISOLATION_PHRASES):
            return {
                "emotion": "sadness",
                "confidence": 1.0,
                "risk_flag": True,
                "tone": EMOTION_META[0]["tone"],
            }

        self._load()
        # Cap text length to avoid token explosion
        inputs = self._tokenizer(
            text[:512],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()

        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())
        label_name = LABEL_TO_NAME.get(top_idx, "uncertain")

        if confidence < threshold:
            return {
                "emotion": "uncertain",
                "confidence": round(confidence, 4),
                "risk_flag": False,
                "tone": "Open, curious, non-assumptive",
            }

        # We do NOT set risk_flag solely based on sadness or fear.
        # Only explicit intent classification or hardcoded crisis phrases should trigger crisis routing.
        risk_flag = False

        return {
            "emotion": label_name,
            "confidence": round(confidence, 4),
            "risk_flag": risk_flag,
            "tone": EMOTION_META.get(top_idx, {}).get("tone", "Gentle, listening"),
        }


# Global single instance
emotion_classifier = EmotionClassifier()
