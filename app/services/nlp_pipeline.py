import os
import re
import time
import random
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.services.language import language_detector
from app.services.emotion import emotion_classifier
from app.services.intent import intent_classifier
from app.services.rag import rag_service
from app.services.session import SessionMemory
from app.services.crisis import get_hotline, CRISIS_RESOURCES_TEMPLATE

# Setup standard logger
logger = logging.getLogger("app_logger")

THERAPIST_BASE_PROMPT = """
You are a warm, deeply empathetic licensed mental health therapist.
Your core principles — never break these:
1. FEEL FIRST, ADVISE SECOND.
2. YOU ARE AFFECTED BY WHAT THEY SHARE.
3. USE THEIR EXACT WORDS.
4. NEVER MINIMIZE.
5. ONE QUESTION AT THE END.
6. LENGTH AND TONE: Keep your response warm, conversational, and concise (1 to 2 short paragraphs).
7. LANGUAGE & CULTURAL STYLE:
    - Always respond in the exact language the user used.
    - If the user writes in Arabic, respond in warm, natural Egyptian Arabic (عامية مصرية بسيطة وواضحة).
    - Use light, appropriate emojis when they naturally fit (💛 🤍 🌷 🫂 😊 💙). Never overuse them in crisis.
8. CRISIS HANDLING: Begin with highly supportive, encouraging, and deeply empathetic words, then attach resources at the end.
"""

FALLBACK_GENERAL = (
    "I am having a little trouble reaching my full resources right now, "
    "but I am here and I am listening. Can you tell me a little more about what brought you here today?"
)

# ── QUICK-RESPONSE SYSTEM ──
_QUICK_PATTERNS = {
    "greeting": {
        "en": ["hi", "hello", "hey", "heyy", "heyyy", "howdy", "yo", "good morning", "good afternoon", "good evening", "whats up", "how are you", "how are u"],
        "ar": ["مرحبا", "مرحبه", "اهلا", "أهلا", "هلا", "السلام عليكم", "صباح الخير", "مساء الخير", "كيف حالك", "كيفك"]
    },
    "gratitude": {
        "en": ["thank you", "thanks", "thank u", "thx", "thanks a lot", "thank you so much", "appreciate it"],
        "ar": ["شكرا", "شكراً", "شكرا لك", "يعطيك العافية", "تسلم", "جزاك الله خيرا"]
    },
    "goodbye": {
        "en": ["bye", "goodbye", "good bye", "see you", "see ya", "take care"],
        "ar": ["مع السلامة", "باي", "في أمان الله", "إلى اللقاء", "سلام"]
    },
    "out_of_scope": {
        "en": ["whats the weather", "tell me a joke", "what time is it", "write code", "how to code", "weather today", "recipe for"],
        "ar": ["كيف الطقس", "كم الساعة", "احكيلي نكتة", "اكتب كود", "برمجة", "طريقة عمل"]
    }
}

_QUICK_RESPONSES = {
    "greeting": {
        "en": ["Hello! 😊 I'm really glad you're here. This is a safe space. What's on your mind today?"],
        "ar": ["أهلًا بيك! 😊 مجرد إنك قررت تتكلم خطوة مهمة وشجاعة. احكي براحتك، وأنا هسمعك من غير أي حكم أو ضغط. 🤗"]
    },
    "gratitude": {
        "en": ["You're so welcome! 💛 Remember, I'm always here whenever you need to talk."],
        "ar": ["العفو! 😊 إنت أظهرت قوة حقيقية بإنك انفتحت وحكيت. اعتني بنفسك، ولا تتردد ترجع في أي وقت. 💛"]
    },
    "goodbye": {
        "en": ["Take care of yourself! 💛 You're not alone in this."],
        "ar": ["اعتني بنفسك! 💛 تذكر، أنا هنا وقت ما تحتاج تحكي في أي وقت. ما إنت لوحدك. مع السلامة! 😊"]
    },
    "out_of_scope": {
        "en": ["I wish I could help with that! 😊 My expertise is specifically in mental health support."],
        "ar": ["أقدر فضولك! 😊 أنا متخصص في دعم الصحة النفسية والعاطفية، وما أقدر أساعدك بهالموضوع. بس لو شايل هم في قلبك أنا هسمعك. 💛"]
    }
}

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)
    return " ".join(text.split())

def _detect_quick_response(text: str) -> Optional[Tuple[str, str]]:
    normalized = _normalize(text)
    if not normalized:
        return None
    for category, lang_map in _QUICK_PATTERNS.items():
        for lang, patterns in lang_map.items():
            if normalized in patterns or any(normalized.startswith(pat + " ") for pat in patterns):
                return category, lang
    return None

class NLPPipeline:
    """Consolidated orchestration pipeline for intent classification, RAG retrieval, and generation."""
    def __init__(self):
        self._groq_client = None

    def _get_groq_client(self):
        if not self._groq_client:
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set.")
            self._groq_client = Groq(api_key=GROQ_API_KEY)
        return self._groq_client

    def _call_therapist_llm(self, query: str, prompt: str, history: list) -> str:
        client = self._get_groq_client()
        messages = [{"role": "system", "content": prompt}]
        # Append sliding conversation history
        if history:
            messages.extend(history[-12:])
        messages.append({"role": "user", "content": query})

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq Chat Completion Error: {e}", exc_info=True)
            return FALLBACK_GENERAL

    def run(self, query: str, session: SessionMemory, country: str = "United States") -> Dict[str, Any]:
        t_start = time.time()
        
        prior_crisis = session.prior_crisis
        history = session.get_history()

        # 1. Fast-path check for simple inputs (sub-1ms)
        quick_match = _detect_quick_response(query)
        if quick_match and not prior_crisis:
            category, lang = quick_match
            response_text = random.choice(_QUICK_RESPONSES[category][lang])
            
            session.add_turn(
                user_message=query,
                assistant_response=response_text,
                emotion="joy" if category != "out_of_scope" else "surprise",
                emotion_conf=0.9,
                language=lang,
                intent=category,
                crisis_flag=False
            )
            
            return {
                "answer": response_text,
                "sources": [],
                "emotion": "joy" if category != "out_of_scope" else "surprise",
                "emotion_conf": 0.9,
                "language": lang,
                "intent": category,
                "crisis_flag": False,
                "latency_ms": round((time.time() - t_start) * 1000, 2),
                "rag_scores": []
            }

        # 2. Run Language Detection & Emotion Classification
        lang_res = language_detector.detect(query)
        lang_code = lang_res["prediction"]  # "en" or "ar"
        if lang_code not in ("en", "ar"):
            lang_code = "en"
        
        emotion_res = emotion_classifier.classify(query)
        emotion = emotion_res["emotion"]
        emotion_conf = emotion_res["confidence"]
        
        # 3. Intent Classification
        intent_res = intent_classifier.classify(
            text=query,
            detected_emotion=emotion,
            detected_language=lang_res["lang_name"]
        )
        intent = intent_res.get("intent", "asking_mental_health_question")
        crisis_flag = intent_res.get("crisis_flag", False) or emotion_res.get("risk_flag", False)
        
        # 4. Out-of-Scope Fallback handling
        if intent == "out_of_scope" and not crisis_flag:
            response_text = random.choice(_QUICK_RESPONSES["out_of_scope"][lang_code])
            session.add_turn(
                user_message=query,
                assistant_response=response_text,
                emotion=emotion,
                emotion_conf=emotion_conf,
                language=lang_code,
                intent="out_of_scope",
                crisis_flag=False
            )
            return {
                "answer": response_text,
                "sources": [],
                "emotion": emotion,
                "emotion_conf": emotion_conf,
                "language": lang_code,
                "intent": "out_of_scope",
                "crisis_flag": False,
                "latency_ms": round((time.time() - t_start) * 1000, 2),
                "rag_scores": []
            }

        # 5. RAG context retrieval
        chunks = []
        rag_scores = []
        if intent == "asking_mental_health_question" or crisis_flag:
            chunks = rag_service.retrieve_and_rerank(query, emotion=emotion)
            rag_scores = [c["similarity"] for c in chunks]

        # 6. Build prompt and invoke LLM
        prompt_sections = [THERAPIST_BASE_PROMPT]
        
        # Crisis warning injection
        if crisis_flag:
            hotline_info = get_hotline(country)
            prompt_sections.append(
                "⚠ CRISIS CONTEXT ACTIVE\nInclude crisis helpline details natively at the end:\n" +
                CRISIS_RESOURCES_TEMPLATE[lang_code].format(**hotline_info)
            )
        elif prior_crisis:
            prompt_sections.append(
                "⚠ PRIOR CRISIS CONTEXT: The user recently expressed thoughts of self-harm. Maintain extra warmth, safety, and gentleness."
            )
        
        if emotion:
            prompt_sections.append(f"Detected user emotion: {emotion} (Tone adjustment direction: {emotion_res.get('tone', '')})")
            
        if lang_code and lang_code != "en":
            prompt_sections.append(f"Response language direction: Respond natively in {lang_code}.")

        if chunks:
            clinical_contexts = "\n".join([f"- Context: {c['context']}\n- Response Guidance: {c['response']}" for c in chunks[:3]])
            prompt_sections.append(f"Clinical counseling knowledge context:\n{clinical_contexts}")

        full_prompt = "\n\n".join(prompt_sections)
        response_text = self._call_therapist_llm(query, full_prompt, history)

        # 7. Append Crisis Resources to final answer if LLM failed to attach it
        if crisis_flag and "https://www.befrienders.org" not in response_text:
            hotline_info = get_hotline(country)
            response_text += CRISIS_RESOURCES_TEMPLATE[lang_code].format(**hotline_info)

        # Update Session history
        session.add_turn(
            user_message=query,
            assistant_response=response_text,
            emotion=emotion,
            emotion_conf=emotion_conf,
            language=lang_code,
            intent=intent,
            crisis_flag=crisis_flag,
            topics=[t for c in chunks for t in c.get("topics", [])]
        )

        return {
            "answer": response_text,
            "sources": [{"excerpt": c["context"][:100] + "...", "similarity": c["similarity"]} for c in chunks],
            "emotion": emotion,
            "emotion_conf": emotion_conf,
            "language": lang_code,
            "intent": intent,
            "crisis_flag": crisis_flag,
            "latency_ms": round((time.time() - t_start) * 1000, 2),
            "rag_scores": rag_scores
        }

# Global pipeline instance
nlp_pipeline = NLPPipeline()
