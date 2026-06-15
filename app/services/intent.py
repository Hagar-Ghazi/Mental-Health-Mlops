import json
import joblib
from typing import Dict, Any, Optional
import google.generativeai as genai
from app.config import INTENT_ARTIFACTS_DIR, GEMINI_API_KEY, GEMINI_MODEL

class IntentClassifier:
    """Lazy loader and wrapper for intent classification and safety filtering using Google Gemini."""
    def __init__(self):
        self._crisis_signals = None
        self._system_prompt = None
        self._model = None
        self._is_loaded = False

    def _load(self):
        if not self._is_loaded:
            config_path = INTENT_ARTIFACTS_DIR / "crisis_config.joblib"
            prompt_path = INTENT_ARTIFACTS_DIR / "system_prompt.txt"
            
            if not config_path.exists():
                raise FileNotFoundError(f"Crisis config not found at {config_path}")
            if not prompt_path.exists():
                raise FileNotFoundError(f"System prompt not found at {prompt_path}")

            config = joblib.load(config_path)
            self._crisis_signals = config["crisis_signals"]

            with open(prompt_path, "r", encoding="utf-8") as f:
                self._system_prompt = f.read()

            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not configured in environment variables.")
                
            genai.configure(api_key=GEMINI_API_KEY)
            self._model = genai.GenerativeModel(
                GEMINI_MODEL,
                system_instruction=self._system_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=300,
                    response_mime_type="application/json"
                )
            )
            self._is_loaded = True

    def has_crisis_signals(self, text: str) -> bool:
        self._load()
        text_lower = text.lower().strip()
        return any(signal in text_lower for signal in self._crisis_signals)

    def classify(self, text: str, detected_emotion: Optional[str] = None, detected_language: Optional[str] = None) -> Dict[str, Any]:
        self._load()
        
        # Hardcoded crisis check (Immediate redirection bypass)
        text_lower = text.lower().strip()
        if any(signal in text_lower for signal in self._crisis_signals):
            return {
                "intent": "asking_mental_health_question",
                "routing": "rag",
                "crisis_flag": True,
                "response_style": "crisis_intervention",
                "confidence": "high"
            }

        try:
            # Enrich context to assist LLM classification
            context_parts = [f"User message: {text}"]
            if detected_emotion:
                context_parts.append(f"Detected emotion: {detected_emotion}")
            if detected_language:
                context_parts.append(f"Detected language: {detected_language}")
            enriched_message = "\n".join(context_parts)

            response = self._model.generate_content(enriched_message)
            
            result = json.loads(response.text.strip())
            result.setdefault("crisis_flag", False)
            return result

        except Exception as e:
            # Fallback values if API call fails
            return {
                "intent": "asking_mental_health_question",
                "routing": "rag",
                "crisis_flag": False,
                "response_style": "empathetic_support",
                "confidence": "low",
                "error": str(e)
            }

# Global single instance
intent_classifier = IntentClassifier()
