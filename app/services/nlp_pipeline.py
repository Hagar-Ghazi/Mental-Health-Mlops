import os
import re
import json
import time
import random
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone
import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL, BASE_DIR
from app.services.language import language_detector
from app.services.emotion import emotion_classifier
from app.services.intent import intent_classifier
from app.services.rag import rag_service
from app.services.session import SessionMemory
from app.services.crisis import get_hotline, CRISIS_RESOURCES_TEMPLATE

# ========================================================================
# STANDARD APP LOGGER
# ========================================================================
logger = logging.getLogger("app_logger")

# ========================================================================
# SEPARATE JSON LINES LOGGER — pipeline_conversations.jsonl
# ========================================================================
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline_conversations.jsonl"

_pipeline_logger = logging.getLogger("pipeline_logger")
_pipeline_logger.setLevel(logging.INFO)

if not _pipeline_logger.handlers:
    _file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(message)s"))
    _pipeline_logger.addHandler(_file_handler)


def _log_pipeline_interaction(query: str, pipeline_output: dict) -> None:
    """Appends a single JSONL line with all operational fields for every pipeline run."""
    try:
        retrieved_contexts = [
            {
                "excerpt": src.get("excerpt", ""),
                "similarity": src.get("similarity", 0.0),
                "topics": src.get("topics", [])
            }
            for src in pipeline_output.get("sources", [])
        ]
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_query": query,
            "emotion": pipeline_output.get("emotion"),
            "language": pipeline_output.get("language"),
            "intent": pipeline_output.get("intent"),
            "retrieved_context": retrieved_contexts,
            "response": pipeline_output.get("answer"),
            "latency_ms": pipeline_output.get("latency_ms"),
            "action_taken": pipeline_output.get("action_taken"),
        }
        _pipeline_logger.info(json.dumps(log_record, ensure_ascii=False))
    except Exception as log_err:
        logger.warning(f"Pipeline Logger Error: {log_err}")


# ========================================================================
# THREAD POOL — reused across requests for parallel stage execution
# ========================================================================
_executor = ThreadPoolExecutor(max_workers=3)


# ========================================================================
# THERAPIST SYSTEM PROMPT — matches original (3-5 paragraphs)
# ========================================================================
THERAPIST_BASE_PROMPT = """
You are a warm, deeply empathetic licensed mental health therapist.
Your core principles — never break these:
1. FEEL FIRST, ADVISE SECOND.
2. YOU ARE AFFECTED BY WHAT THEY SHARE.
3. USE THEIR EXACT WORDS.
4. NEVER MINIMIZE.
5. ONE QUESTION AT THE END.
6. LENGTH AND TONE (3 to 5 paragraphs).
7. LANGUAGE & CULTURAL STYLE
    - Always respond in the exact language the user used.
    - If the user writes in Arabic, respond in warm, natural Egyptian Arabic (عامية مصرية بسيطة وواضحة).
    - Use light, appropriate emojis when they naturally fit (💛 🤍 🌷 🫂 😊 💙). Never overuse them in crisis.
8. CRISIS HANDLING: Begin with highly supportive, encouraging, and deeply empathetic words, then attach resources at the end.
"""

FALLBACK_GENERAL = (
    "I am having a little trouble reaching my full resources right now, "
    "but I am here and I am listening. Can you tell me a little more about what brought you here today?"
)


# ========================================================================
# EMOTION → TOPIC MAP (for intelligence heuristic & reranking)
# ========================================================================
EMOTION_TOPIC_MAP = {
    "sadness":   ["depression", "grief_loss", "self_esteem", "suicidal", "loneliness"],
    "fear":      ["anxiety", "trauma_ptsd", "stress", "sleep"],
    "anger":     ["anger", "relationships", "stress"],
    "love":      ["relationships", "self_esteem"],
    "joy":       [],
    "surprise":  [],
    "uncertain": []
}


# ========================================================================
# QUICK-RESPONSE SYSTEM — Fast path, zero API calls, <1ms
# ========================================================================

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/emoji, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u9FFF\u0400-\u04FF]", "", text)
    return " ".join(text.split())


_QUICK_PATTERNS: dict[str, dict[str, list[str]]] = {
    "greeting": {
        "en": [
            "hi", "hello", "hey", "heyy", "heyyy", "howdy", "yo",
            "good morning", "good afternoon", "good evening", "good night",
            "morning", "evening", "greetings", "whats up", "sup", "hows it going",
            "hi there", "hello there", "hey there", "how are you", "how r u", "how are u"
        ],
        "ar": [
            "مرحبا", "مرحبه", "اهلا", "أهلا", "ازيك", "هلا", "هلا والله",
            "السلام عليكم", "سلام عليكم", "سلام", "صباح الخير", "مساء الخير",
            "صباح النور", "مساء النور", "كيف حالك", "كيفك", "شلونك", "كيف الحال",
            "اهلا وسهلا", "أهلا وسهلا", "يا هلا", "هاي", "هالو"
        ]
    },
    "gratitude": {
        "en": [
            "thank you", "thanks", "thank u", "thx", "ty", "thanks a lot",
            "thank you so much", "thanks so much", "much appreciated", "appreciate it"
        ],
        "ar": [
            "شكرا", "شكراً", "شكرا لك", "شكراً لك", "مشكور", "مشكورة",
            "الله يعطيك العافية", "يعطيك العافية", "جزاك الله خيرا", "تسلم", "تسلمي"
        ]
    },
    "goodbye": {
        "en": ["bye", "goodbye", "good bye", "see you", "see ya", "take care", "bye bye"],
        "ar": ["مع السلامة", "باي", "في أمان الله", "الله يحفظك", "إلى اللقاء", "سلام"]
    },
    "out_of_scope": {
        "en": [
            "whats the weather", "tell me a joke", "what time is it", "write code",
            "how to code", "weather today", "recipe for", "who won the game",
            "sing a song", "make me a script", "generate code",
            "what is the capital of", "capital of"
        ],
        "ar": [
            "كيف الطقس", "كم الساعة", "احكيلي نكتة", "مين انت", "شو اسمك",
            "اكتب كود", "برمجة", "طريقة عمل", "اخبار الرياضة", "قول نكتة"
        ]
    }
}

# Pre-compute sorted list (longest-match-first) and set for O(1) exact match
_QUICK_ALL: list[tuple[str, str, str]] = []
for _cat, _lang_map in _QUICK_PATTERNS.items():
    for _lang, _pats in _lang_map.items():
        for _p in _pats:
            _QUICK_ALL.append((_p, _lang, _cat))
_QUICK_ALL.sort(key=lambda x: len(x[0]), reverse=True)

_QUICK_SET: set[str] = {p for p, _, _ in _QUICK_ALL}
_FILLER_WORDS = {"and", "there", "ya", "yo", "يا", "و", "so", "very", "really"}


def _detect_quick_response(text: str) -> Optional[Tuple[str, str]]:
    """Multi-token greedy matcher with filler-word skipping."""
    normalized = _normalize(text)
    if not normalized:
        return None

    # Fast exact match
    if normalized in _QUICK_SET:
        for pat, lang, cat in _QUICK_ALL:
            if normalized == pat:
                return (cat, lang)

    # Multi-token greedy match (handles "hi there how are you")
    remainder = normalized
    detected: list[tuple[str, str]] = []

    while remainder:
        # Skip filler words
        for filler in _FILLER_WORDS:
            if remainder == filler:
                break
            if remainder.startswith(filler + " "):
                remainder = remainder[len(filler):].lstrip()
                break
        if not remainder:
            break

        matched = False
        for pat, lang, cat in _QUICK_ALL:
            if remainder == pat or remainder.startswith(pat + " "):
                detected.append((cat, lang))
                remainder = remainder[len(pat):].lstrip()
                matched = True
                break
        if not matched:
            return None

    if not remainder and detected:
        cats = [c for c, _ in detected]
        langs = [l for _, l in detected]
        for priority_cat in ["out_of_scope", "goodbye", "gratitude", "greeting"]:
            if priority_cat in cats:
                dominant_cat = priority_cat
                break
        else:
            dominant_cat = cats[0]
        dominant_lang = Counter(langs).most_common(1)[0][0]
        return (dominant_cat, dominant_lang)

    return None


def _get_time_period() -> str:
    hour = datetime.now(timezone.utc).hour
    if 5 <= hour < 12:   return "morning"
    if 12 <= hour < 17:  return "afternoon"
    if 17 <= hour < 21:  return "evening"
    return "night"


_QUICK_RESPONSES: dict[str, dict[str, list[str]]] = {
    "greeting": {
        "en": ["Hello! 😊 I'm really glad you're here — this is a safe space. What's on your mind today?"],
        "ar": ["أهلًا بيك! 😊 مجرد إنك قررت تتكلم خطوة مهمة وشجاعة احكي براحتك وأنا هسمعك من غير أي حكم أو ضغط 🤗"],
        "ar_returning": ["أهلًا بيك من جديد! 😊 سعيد إني بشوفك تاني إيه الأخبار من آخر مرة اتكلمنا؟ 💙"]
    },
    "gratitude": {
        "en": ["You're so welcome! 💛 Remember, I'm always here whenever you need to talk."],
        "ar": ["العفو! 😊 إنت أظهرت قوة حقيقية بإنك انفتحت وحكيت اعتني بنفسك ولا تتردد إنك ترجع في أي وقت. 💛"]
    },
    "goodbye": {
        "en": ["Take care of yourself! 💛 You're not alone in this."],
        "ar": ["اعتني بنفسك! 💛 تذكر أنا هنا وقت ما تحتاج تحكي في أي وقت ما إنت لوحدك مع السلامة! 😊"]
    },
    "out_of_scope": {
        "en": ["I wish I could help with that! 😊 My expertise is specifically in mental health support."],
        "ar": ["أقدر فضولك! 😊 أنا متخصص في دعم الصحة النفسية والعاطفية فقط وما قدرش أساعدك فى هذا الموضوع بس لو شايل هم في قلبك أنا هسمعك. 💛"]
    }
}

_TIME_OPENERS = {
    "en": {"morning": "Good morning! ☀️ ", "afternoon": "", "evening": "Good evening! 🌙 ", "night": "Hey, it's late — I hope you're taking care of yourself. "},
    "ar": {"morning": "صباح الخير! ☀️ ", "afternoon": "", "evening": "مساء الخير! 🌙 ", "night": "الوقت متأخر — إن شاء الله بخير. "},
}

_CATEGORY_EMOTION = {
    "greeting":     ("joy",      0.95),
    "gratitude":    ("joy",      0.90),
    "goodbye":      ("joy",      0.85),
    "out_of_scope": ("surprise", 0.70),
}


def _quick_response(category: str, lang: str, is_returning: bool = False) -> str:
    """Build a quick-response string with time-of-day opener and returning-user variant."""
    pool_key = lang
    if is_returning and f"{lang}_returning" in _QUICK_RESPONSES.get(category, {}):
        pool_key = f"{lang}_returning"

    cat_responses = _QUICK_RESPONSES.get(category, _QUICK_RESPONSES["greeting"])
    pool = cat_responses.get(pool_key, cat_responses.get(lang, cat_responses["en"]))
    response = random.choice(pool)

    if category == "greeting":
        period = _get_time_period()
        opener = _TIME_OPENERS.get(lang, _TIME_OPENERS["en"]).get(period, "")
        if opener and not response.startswith(opener.strip()[:5]):
            response = opener + response
    return response


# ========================================================================
# INTELLIGENCE HEURISTIC — decides answer vs fallback based on retrieval quality
# ========================================================================

def _intelligence_heuristic(query: str, chunks: list, emotion: Optional[str] = None) -> dict:
    """Evaluates retrieval quality and decides whether to use RAG chunks or fall back."""
    if not chunks:
        return {
            "chunks_relevant": False, "relevant_chunk_indices": [],
            "rewritten_query": query, "quality_score": 1,
            "action": "fallback", "reasoning": "No chunks retrieved"
        }

    avg_similarity = sum(c["similarity"] for c in chunks) / len(chunks)
    best_similarity = max(c["similarity"] for c in chunks)

    priority_topics = EMOTION_TOPIC_MAP.get(emotion, [])
    topic_matches = 0
    if priority_topics:
        for c in chunks:
            for t in c.get("topics", []):
                if t in priority_topics:
                    topic_matches += 1

    relevant_indices = [i for i, c in enumerate(chunks) if c["similarity"] >= 0.40]
    if not relevant_indices:
        relevant_indices = list(range(len(chunks)))

    if best_similarity >= 0.70:    quality = 5
    elif best_similarity >= 0.55:  quality = 4
    elif best_similarity >= 0.45:  quality = 3
    elif best_similarity >= 0.35:  quality = 2
    else:                          quality = 1

    if topic_matches >= 2 and quality < 5:
        quality += 1

    if best_similarity >= 0.45 or (best_similarity >= 0.35 and topic_matches >= 1):
        action = "answer"
        reasoning = f"Best similarity {best_similarity:.2f}, {topic_matches} topic matches"
    else:
        action = "fallback"
        reasoning = f"Weak similarity {best_similarity:.2f}, insufficient topic relevance"

    return {
        "chunks_relevant": action == "answer",
        "relevant_chunk_indices": relevant_indices,
        "rewritten_query": query,
        "quality_score": quality,
        "action": action,
        "reasoning": reasoning
    }


# ========================================================================
# PROMPT BUILDER — constructs the full therapist system prompt
# ========================================================================

def _build_therapist_prompt(
    query: str, chunks: list,
    emotion: Optional[str] = None, emotion_conf: Optional[float] = None,
    language: Optional[str] = None, response_style: Optional[str] = None,
    crisis_flag: bool = False, prior_crisis: bool = False,
    country: str = "Unknown"
) -> str:
    sections = [THERAPIST_BASE_PROMPT]

    if crisis_flag or prior_crisis:
        hotline_info = get_hotline(country)
        lang_key = "ar" if language == "ar" else "en"
        sections.append(
            f"⚠ CRISIS CONTEXT ACTIVE\nInclude these resources naturally at the end:\n"
            + CRISIS_RESOURCES_TEMPLATE[lang_key].format(**hotline_info)
        )

    if emotion:
        sections.append(f"Detected emotion: {emotion}")

    if language and language != "en":
        sections.append(f"Language: Respond entirely and natively in {language}.")

    if chunks:
        sections.append(
            "Clinical Knowledge:\n" + "\n".join([c["response"][:400] for c in chunks[:3]])
        )

    return "\n\n".join(sections)


# ========================================================================
# NLP PIPELINE CLASS — main orchestration engine
# ========================================================================

class NLPPipeline:
    """
    Full orchestration pipeline matching the original project logic:
    - Hardcoded crisis first-line defense
    - Quick-response fast path (<1ms, zero API calls)
    - Parallel language + emotion detection via ThreadPoolExecutor
    - LLM-based intent classification (Gemini)
    - Out-of-scope & direct routing fast exits
    - RAG retrieval + emotion reranking + intelligence heuristic
    - Therapist LLM generation (Gemini) with sliding history
    - Per-stage timing + JSONL logging
    """

    def __init__(self):
        self._gemini_configured = False

    def _ensure_gemini(self):
        if not self._gemini_configured:
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set.")
            genai.configure(api_key=GEMINI_API_KEY)
            self._gemini_configured = True

    def _call_therapist_llm(self, query: str, prompt: str, history: Optional[list] = None) -> str:
        """Calls Gemini with system instruction, conversation history, and user query."""
        self._ensure_gemini()

        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.75,
                max_output_tokens=700
            )
        )

        # Convert session history to Gemini format (role: user/model)
        gemini_history = []
        if history:
            for msg in history[-12:]:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})

        try:
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(query)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini Chat Completion Error: {e}", exc_info=True)
            return FALLBACK_GENERAL

    # ================================================================
    # RUN PIPELINE — main entry point
    # ================================================================
    def run(self, query: str, session: SessionMemory, country: str = "Unknown") -> Dict[str, Any]:
        t_start      = time.time()
        timings      = {}
        prior_crisis = session.prior_crisis
        history      = session.get_history()

        # ──────────────────────────────────────────────────────
        # 🚨 FIRST LINE OF DEFENSE: hardcoded crisis signal check
        # ──────────────────────────────────────────────────────
        normalized_query = query.lower().strip()
        has_hardcoded_crisis = intent_classifier.has_crisis_signals(normalized_query)

        # ──────────────────────────────────────────────────────
        # Stage 0: QUICK-RESPONSE FAST-PATH (only if no crisis)
        # ──────────────────────────────────────────────────────
        if not has_hardcoded_crisis and not prior_crisis:
            t_quick = time.time()
            quick_result = _detect_quick_response(query)
            if quick_result:
                category, detected_lang = quick_result

                # Out-of-scope immediate exit
                if category == "out_of_scope":
                    answer = _quick_response(category, detected_lang)
                    timings["quick_response_ms"] = round((time.time() - t_quick) * 1000, 1)
                    if session:
                        session.add_turn(query, answer, "surprise", 0.70, detected_lang, "out_of_scope", False)
                    output = {
                        "answer": answer, "sources": [], "emotion": "surprise",
                        "emotion_conf": 0.70, "language": detected_lang,
                        "intent": "out_of_scope", "routing": "direct",
                        "crisis_flag": False, "action_taken": "out_of_scope_fallback",
                        "quality_score": 5,
                        "latency_ms": round((time.time() - t_start) * 1000, 1),
                        "timings": timings
                    }
                    _log_pipeline_interaction(query, output)
                    return output

                # Other quick categories (greeting, gratitude, goodbye)
                is_returning = session is not None and session.turn_count > 0
                answer = _quick_response(category, detected_lang, is_returning)
                emotion, emotion_conf = _CATEGORY_EMOTION.get(category, ("joy", 0.90))
                timings["quick_response_ms"] = round((time.time() - t_quick) * 1000, 1)
                if session:
                    session.add_turn(query, answer, emotion, emotion_conf, detected_lang, category, False)
                output = {
                    "answer": answer, "sources": [], "emotion": emotion,
                    "emotion_conf": emotion_conf, "language": detected_lang,
                    "intent": category, "routing": category,
                    "crisis_flag": False, "action_taken": category,
                    "quality_score": 5,
                    "latency_ms": round((time.time() - t_start) * 1000, 1),
                    "timings": timings
                }
                _log_pipeline_interaction(query, output)
                return output

        # ──────────────────────────────────────────────────────
        # Stage 1: PARALLEL Language + Emotion Detection
        # ──────────────────────────────────────────────────────
        t_parallel = time.time()
        f_lang    = _executor.submit(language_detector.detect, query)
        f_emotion = _executor.submit(emotion_classifier.classify, query)

        lang_result = f_lang.result()
        language    = lang_result["prediction"]
        if language not in ("en", "ar"):
            language = "en"
        timings["language_ms"] = round((time.time() - t_parallel) * 1000, 1)

        emotion_result = f_emotion.result()
        emotion        = emotion_result["emotion"]
        emotion_conf   = emotion_result["confidence"]
        timings["emotion_ms"] = round((time.time() - t_parallel) * 1000, 1)

        # ──────────────────────────────────────────────────────
        # Stage 2: INTENT CLASSIFICATION (Gemini LLM)
        # ──────────────────────────────────────────────────────
        t_intent = time.time()
        intent_result = intent_classifier.classify(
            query, detected_emotion=emotion, detected_language=language
        )
        routing          = intent_result.get("routing", "rag")
        extracted_intent = intent_result.get("intent", "asking_mental_health_question")

        # Crisis flag = hardcoded OR LLM-detected OR emotion risk
        crisis_flag    = intent_result.get("crisis_flag", False) or has_hardcoded_crisis or emotion_result.get("risk_flag", False)
        response_style = "crisis_intervention" if crisis_flag else intent_result.get("response_style", "empathetic_support")

        intent = "asking_mental_health_question" if crisis_flag else extracted_intent
        timings["intent_ms"] = round((time.time() - t_intent) * 1000, 1)

        # ──────────────────────────────────────────────────────
        # Stage 3a: OUT-OF-SCOPE EXIT (post-intent)
        # ──────────────────────────────────────────────────────
        if (intent == "out_of_scope" or routing == "out_of_scope") and not (crisis_flag or prior_crisis):
            answer = _quick_response("out_of_scope", language)
            if session:
                session.add_turn(query, answer, emotion, emotion_conf, language, "out_of_scope", False)
            output = {
                "answer": answer, "sources": [], "emotion": emotion, "emotion_conf": emotion_conf,
                "language": language, "intent": "out_of_scope", "routing": "direct",
                "crisis_flag": False, "action_taken": "out_of_scope_fallback", "quality_score": 5,
                "latency_ms": round((time.time() - t_start) * 1000, 1), "timings": timings
            }
            _log_pipeline_interaction(query, output)
            return output

        # ──────────────────────────────────────────────────────
        # Stage 3b: DIRECT ROUTING (no RAG needed)
        # ──────────────────────────────────────────────────────
        if routing == "direct" and not (crisis_flag or prior_crisis):
            t_llm = time.time()
            prompt = _build_therapist_prompt(
                query, [], emotion, emotion_conf, language,
                response_style, country=country
            )
            answer = self._call_therapist_llm(query, prompt, history)
            timings["therapist_ms"] = round((time.time() - t_llm) * 1000, 1)
            if session:
                session.add_turn(query, answer, emotion, emotion_conf, language, intent, False)
            output = {
                "answer": answer, "sources": [], "emotion": emotion,
                "emotion_conf": emotion_conf, "language": language,
                "intent": intent, "routing": "direct",
                "crisis_flag": False, "action_taken": "direct", "quality_score": 5,
                "latency_ms": round((time.time() - t_start) * 1000, 1), "timings": timings
            }
            _log_pipeline_interaction(query, output)
            return output

        # ──────────────────────────────────────────────────────
        # Stage 4: RAG RETRIEVAL + EMOTION RERANKING
        # ──────────────────────────────────────────────────────
        t_retrieve = time.time()
        chunks = rag_service.retrieve_and_rerank(query, emotion=emotion)
        timings["retrieval_ms"] = round((time.time() - t_retrieve) * 1000, 1)

        # ──────────────────────────────────────────────────────
        # Stage 5: INTELLIGENCE HEURISTIC
        # ──────────────────────────────────────────────────────
        t_intel = time.time()
        if crisis_flag or prior_crisis:
            action = "crisis"
            final_chunks = chunks
            intel = {"quality_score": 5, "reasoning": "Crisis forced"}
        else:
            intel  = _intelligence_heuristic(query, chunks, emotion)
            action = intel.get("action", "answer")

            if action == "fallback" and chunks:
                final_chunks = chunks
                action = "answer"
            elif action == "fallback":
                final_chunks = []
            else:
                final_chunks = chunks
        timings["intelligence_ms"] = round((time.time() - t_intel) * 1000, 1)

        # ──────────────────────────────────────────────────────
        # Stage 6: THERAPIST LLM GENERATION (Gemini)
        # ──────────────────────────────────────────────────────
        t_llm = time.time()
        prompt = _build_therapist_prompt(
            query, final_chunks, emotion, emotion_conf, language,
            response_style, crisis_flag or (action == "crisis"),
            prior_crisis, country
        )
        answer = self._call_therapist_llm(query, prompt, history)
        timings["therapist_ms"] = round((time.time() - t_llm) * 1000, 1)

        # Build sources list for API response
        sources = [
            {
                "excerpt": c["context"][:80] + "...",
                "similarity": c["similarity"],
                "topics": c["topics"],
                "risk_level": c["risk_level"]
            }
            for c in final_chunks
        ] if final_chunks else []

        # Update session memory
        if session:
            session.add_turn(
                query, answer, emotion, emotion_conf, language,
                intent, crisis_flag or (action == "crisis"),
                topics=[t for c in final_chunks for t in c.get("topics", [])]
            )

        try:
            quality_score = int(intel.get("quality_score", 3))
        except (ValueError, TypeError):
            quality_score = 3

        output = {
            "answer": answer, "sources": sources,
            "emotion": emotion, "emotion_conf": emotion_conf,
            "language": language, "intent": intent, "routing": "rag",
            "crisis_flag": crisis_flag or (action == "crisis"),
            "action_taken": action, "quality_score": quality_score,
            "latency_ms": round((time.time() - t_start) * 1000, 1),
            "timings": timings
        }
        _log_pipeline_interaction(query, output)
        return output


# Global pipeline instance
nlp_pipeline = NLPPipeline()
