import joblib
from typing import Dict, Any
from app.config import LANGUAGE_DETECTOR_PATH


class LanguageDetector:
    """Lazy loader and wrapper for the scikit-learn Language Detector model."""

    def __init__(self):
        self._model = None
        self._vectorizer = None
        self._language_meta = None
        self._confidence_threshold = None
        self._short_threshold = None
        self._is_loaded = False

    def _load(self):
        if not self._is_loaded:
            if not LANGUAGE_DETECTOR_PATH.exists():
                raise FileNotFoundError(
                    f"Language detector model not found at {LANGUAGE_DETECTOR_PATH}"
                )

            artifacts = joblib.load(LANGUAGE_DETECTOR_PATH)
            self._model = artifacts["model"]
            self._vectorizer = artifacts["vectorizer"]
            self._language_meta = artifacts["language_meta"]
            self._confidence_threshold = artifacts.get("confidence_threshold", 0.70)
            self._short_threshold = artifacts.get("short_threshold", 8)
            self._is_loaded = True

    def detect(self, text: str) -> Dict[str, Any]:
        self._load()
        clean = " ".join(text.strip().split())
        is_short = len(clean) <= self._short_threshold

        features = self._vectorizer.transform([clean])
        prediction = self._model.predict(features)[0]
        proba = self._model.predict_proba(features)[0]
        confidence = float(proba.max())

        lang_name = (
            self._language_meta[prediction][0]
            if prediction in self._language_meta
            else str(prediction)
        )

        return {
            "prediction": prediction,  # e.g. "en", "ar"
            "lang_name": lang_name,  # e.g. "English", "Arabic"
            "confidence": round(confidence, 4),
            "trusted": not (is_short and confidence < self._confidence_threshold),
        }


# Global single instance
language_detector = LanguageDetector()
