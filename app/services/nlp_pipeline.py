# import re
# import json
# import time
# import random
# import logging
# import asyncio
# from collections import Counter
# from typing import Dict, Any, Optional, Tuple
# from datetime import datetime, timezone
# from fastapi.concurrency import run_in_threadpool
# from app.config import GROQ_API_KEY, GROQ_MODEL, BASE_DIR
# from app.services.language import language_detector
# from app.services.emotion import emotion_classifier
# from app.services.intent import intent_classifier
# from app.services.rag import rag_service
# from app.services.session import SessionMemory
# from app.services.crisis import get_hotline, CRISIS_RESOURCES_TEMPLATE

# # ========================================================================
# # STANDARD APP LOGGER
# # ========================================================================
# logger = logging.getLogger("app_logger")

# # ========================================================================
# # SEPARATE JSON LINES LOGGER — pipeline_conversations.jsonl
# # ========================================================================
# LOG_DIR = BASE_DIR / "logs"
# LOG_DIR.mkdir(exist_ok=True)
# LOG_FILE = LOG_DIR / "pipeline_conversations.jsonl"

# _pipeline_logger = logging.getLogger("pipeline_logger")
# _pipeline_logger.setLevel(logging.INFO)

# if not _pipeline_logger.handlers:
#     _file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
#     _file_handler.setFormatter(logging.Formatter("%(message)s"))
#     _pipeline_logger.addHandler(_file_handler)


# def _log_pipeline_interaction(query: str, pipeline_output: dict) -> None:
#     """Appends a single JSONL line with all operational fields for every pipeline run."""
#     try:
#         retrieved_contexts = [
#             {
#                 "excerpt": src.get("excerpt", ""),
#                 "similarity": src.get("similarity", 0.0),
#                 "topics": src.get("topics", []),
#             }
#             for src in pipeline_output.get("sources", [])
#         ]
#         log_record = {
#             "timestamp": datetime.now(timezone.utc).isoformat(),
#             "user_query": query,
#             "emotion": pipeline_output.get("emotion"),
#             "language": pipeline_output.get("language"),
#             "intent": pipeline_output.get("intent"),
#             "retrieved_context": retrieved_contexts,
#             "response": pipeline_output.get("answer"),
#             "latency_ms": pipeline_output.get("latency_ms"),
#             "action_taken": pipeline_output.get("action_taken"),
#         }
#         _pipeline_logger.info(json.dumps(log_record, ensure_ascii=False))
#     except Exception as log_err:
#         logger.warning(f"Pipeline Logger Error: {log_err}")


# # THREAD POOL removed we now use fastapi.concurrency.run_in_threadpool
# # ========================================================================
# # THERAPIST SYSTEM PROMPT — matches original (3-5 paragraphs)
# # ========================================================================
# THERAPIST_BASE_PROMPT = """
# You are a warm, deeply empathetic, and highly conversational mental health supporter.
# Your core principles:
# 1. BE NATURAL AND CONVERSATIONAL: Talk like a caring human, not a robotic textbook therapist. Do not awkwardly repeat the user's exact words back to them.
# 2. MATCH THE TONE: If the user is casual, lighthearted, or asking a simple question (e.g., about a recipe, a hobby, or feeling happy), be friendly, brief, and conversational. Save deep therapeutic reflection ONLY for actual distress or emotional pain.
# 3. NEVER MINIMIZE PAIN: If they share pain, validate it warmly before offering any advice.
# 4. RESPONSE LENGTH AND DETAIL: Talk like a caring, comforting supporter. If the user is sharing actual pain, distress, sadness, or venting about a difficult/complex situation, you MUST write a longer, deeper, and more detailed comforting response (2 to 4 paragraphs) to validate their feelings, explore their situation, and offer deep warmth and empathy. Do NOT write short 1-paragraph answers for serious distress. However, if the user is greeting you or asking a simple, lighthearted question, keep it brief and conversational (1 to 2 short paragraphs).
# 5. GENTLE GUIDANCE: You may ask a natural question to keep the conversation flowing, but only if it makes sense. Do not force a probing psychological question on a casual topic.
# 6. LANGUAGE & CULTURAL STYLE:
#     - Always respond STRICTLY in the single language the user used.
#     - If the user writes in English, respond ENTIRELY in English. Do NOT include any Arabic words, Arabic characters, or Franco-Arabic transliterations (like writing Arabic words using English letters).
#     - If the user writes in Arabic, respond ENTIRELY in simple, warm Egyptian Arabic (عامية مصرية بسيطة ولهجة مصرية عامية دافئة وحقيقية). Do NOT use Modern Standard Arabic (الفصحى) except for extremely common phrases.
#     - NEVER mix languages or provide translations or transliterations in your response. Do NOT use non-Arabic letters or foreign scripts (no English, no Chinese, no Japanese, no French, and no Latin script) when replying to Arabic queries. Write using only standard Arabic script.
#     - EMOJIS & WARMTH: Use warm, comforting emojis and hearts when they naturally fit (e.g., 🫂, 💛, 🤍, 🌷, 🩹, ❤️, 😊) to show emotional warmth, support, and care, especially in emotional contexts or responses that need heart. Never overuse them in a severe crisis, but use them to make your support feel genuinely warm, human, and caring.
# 7. CRISIS HANDLING: If a crisis is detected, prioritize immediate safety, extreme warmth, and direct the user to the provided resources without being pushy.
# """

# FALLBACK_GENERAL = (
#     "I am having a little trouble reaching my full resources right now, "
#     "but I am here and I am listening. Can you tell me a little more about what brought you here today?"
# )


# # ========================================================================
# # EMOTION → TOPIC MAP (for intelligence heuristic & reranking)
# # ========================================================================
# EMOTION_TOPIC_MAP = {
#     "sadness": ["depression", "grief_loss", "self_esteem", "suicidal", "loneliness"],
#     "fear": ["anxiety", "trauma_ptsd", "stress", "sleep"],
#     "anger": ["anger", "relationships", "stress"],
#     "love": ["relationships", "self_esteem"],
#     "joy": [],
#     "surprise": [],
#     "uncertain": [],
# }


# # ========================================================================
# # QUICK-RESPONSE SYSTEM — Fast path, zero API calls, <1ms
# # ========================================================================


# def _normalize(text: str) -> str:
#     """Lowercase, strip punctuation/emoji, collapse whitespace."""
#     text = text.lower().strip()
#     text = re.sub(
#         r"[^\w\s\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u9FFF\u0400-\u04FF]",
#         "",
#         text,
#     )
#     return " ".join(text.split())


# _QUICK_PATTERNS: dict[str, dict[str, list[str]]] = {
#     "greeting": {
#         "en": [
#             "hi",
#             "hello",
#             "hey",
#             "heyy",
#             "heyyy",
#             "howdy",
#             "yo",
#             "good morning",
#             "good afternoon",
#             "good evening",
#             "good night",
#             "morning",
#             "evening",
#             "greetings",
#             "whats up",
#             "sup",
#             "hows it going",
#             "hi there",
#             "hello there",
#             "hey there",
#             "how are you",
#             "how r u",
#             "how are u",
#         ],
#         "ar": [
#             "مرحبا",
#             "مرحبه",
#             "اهلا",
#             "أهلا",
#             "ازيك",
#             "هلا",
#             "هلا والله",
#             "السلام عليكم",
#             "سلام عليكم",
#             "سلام",
#             "صباح الخير",
#             "مساء الخير",
#             "صباح النور",
#             "مساء النور",
#             "كيف حالك",
#             "كيفك",
#             "شلونك",
#             "كيف الحال",
#             "اهلا وسهلا",
#             "أهلا وسهلا",
#             "يا هلا",
#             "هاي",
#             "هالو",
#         ],
#     },
#     "gratitude": {
#         "en": [
#             "thank you",
#             "thanks",
#             "thank u",
#             "thx",
#             "ty",
#             "thanks a lot",
#             "thank you so much",
#             "thanks so much",
#             "much appreciated",
#             "appreciate it",
#         ],
#         "ar": [
#             "شكرا",
#             "شكراً",
#             "شكرا لك",
#             "شكراً لك",
#             "مشكور",
#             "مشكورة",
#             "الله يعطيك العافية",
#             "يعطيك العافية",
#             "جزاك الله خيرا",
#             "تسلم",
#             "تسلمي",
#         ],
#     },
#     "goodbye": {
#         "en": [
#             "bye",
#             "goodbye",
#             "good bye",
#             "see you",
#             "see ya",
#             "take care",
#             "bye bye",
#         ],
#         "ar": ["مع السلامة", "باي", "في أمان الله", "الله يحفظك", "إلى اللقاء", "سلام"],
#     },
#     "out_of_scope": {
#         "en": [
#             "whats the weather",
#             "tell me a joke",
#             "what time is it",
#             "write code",
#             "how to code",
#             "weather today",
#             "recipe for",
#             "who won the game",
#             "sing a song",
#             "make me a script",
#             "generate code",
#             "what is the capital of",
#             "capital of",
#         ],
#         "ar": [
#             "كيف الطقس",
#             "كم الساعة",
#             "احكيلي نكتة",
#             "مين انت",
#             "شو اسمك",
#             "اكتب كود",
#             "برمجة",
#             "طريقة عمل",
#             "اخبار الرياضة",
#             "قول نكتة",
#         ],
#     },
# }

# # Pre-compute sorted list (longest-match-first) and set for O(1) exact match
# _QUICK_ALL: list[tuple[str, str, str]] = []
# for _cat, _lang_map in _QUICK_PATTERNS.items():
#     for _lang, _pats in _lang_map.items():
#         for _p in _pats:
#             _QUICK_ALL.append((_p, _lang, _cat))
# _QUICK_ALL.sort(key=lambda x: len(x[0]), reverse=True)

# _QUICK_SET: set[str] = {p for p, _, _ in _QUICK_ALL}
# _FILLER_WORDS = {"and", "there", "ya", "yo", "يا", "و", "so", "very", "really"}


# def _detect_quick_response(text: str) -> Optional[Tuple[str, str]]:
#     """Multi-token greedy matcher with filler-word skipping."""
#     normalized = _normalize(text)
#     if not normalized:
#         return None

#     # Fast exact match
#     if normalized in _QUICK_SET:
#         for pat, lang, cat in _QUICK_ALL:
#             if normalized == pat:
#                 return (cat, lang)

#     # Multi-token greedy match (handles "hi there how are you")
#     remainder = normalized
#     detected: list[tuple[str, str]] = []

#     while remainder:
#         # Skip filler words
#         for filler in _FILLER_WORDS:
#             if remainder == filler:
#                 break
#             if remainder.startswith(filler + " "):
#                 remainder = remainder[len(filler) :].lstrip()
#                 break
#         if not remainder:
#             break

#         matched = False
#         for pat, lang, cat in _QUICK_ALL:
#             if remainder == pat or remainder.startswith(pat + " "):
#                 detected.append((cat, lang))
#                 remainder = remainder[len(pat) :].lstrip()
#                 matched = True
#                 break
#         if not matched:
#             return None

#     if not remainder and detected:
#         cats = [c for c, _ in detected]
#         langs = [lang for _, lang in detected]
#         for priority_cat in ["out_of_scope", "goodbye", "gratitude", "greeting"]:
#             if priority_cat in cats:
#                 dominant_cat = priority_cat
#                 break
#         else:
#             dominant_cat = cats[0]
#         dominant_lang = Counter(langs).most_common(1)[0][0]
#         return (dominant_cat, dominant_lang)

#     return None


# def _get_time_period() -> str:
#     hour = datetime.now(timezone.utc).hour
#     if 5 <= hour < 12:
#         return "morning"
#     if 12 <= hour < 17:
#         return "afternoon"
#     if 17 <= hour < 21:
#         return "evening"
#     return "night"


# _QUICK_RESPONSES: dict[str, dict[str, list[str]]] = {
#     "greeting": {
#         "en": [
#             "Hello! 😊 I'm really glad you're here this is a safe space what's on your mind today?"
#         ],
#         "ar": [
#             "أهلًا بيك! 😊 مجرد إنك قررت تتكلم خطوة مهمة وشجاعة احكي براحتك وأنا هسمعك من غير أي حكم أو ضغط 🤗"
#         ],
#         "ar_returning": [
#             "أهلًا بيك من جديد 😊 سعيد إني بشوفك تاني إيه الأخبار من آخر مرة اتكلمنا؟ 💙"
#         ],
#     },
#     "gratitude": {
#         "en": [
#             "You're so welcome! 💛 Remember I'm always here whenever you need to talk"
#         ],
#         "ar": [
#             "العفو 😊 إنت أظهرت قوة حقيقية بإنك انفتحت وحكيت اعتني بنفسك ولا تتردد إنك ترجع في أي وقت. 💛"
#         ],
#     },
#     "goodbye": {
#         "en": ["Take care of yourself 💛 You're not alone in this."],
#         "ar": [
#             "اعتني بنفسك 💛وتذكر انا هنا وقت ما تحتاج تتكلم في أي وقت ما إنت مش لوحدك مع السلامة! 😊"
#         ],
#     },
#     "out_of_scope": {
#         "en": [
#             "I wish I could help with that! 😊 My expertise is specifically in mental health support."
#         ],
#         "ar": [
#             "أقدر فضولك! 😊 أنا متخصص في دعم الصحة النفسية والعاطفية فقط وما قدرش أساعدك فى هذا الموضوع بس لو شايل هم في قلبك أنا هسمعك. 💛"
#         ],
#     },
# }

# _TIME_OPENERS = {
#     "en": {
#         "morning": "Good morning! ☀️ ",
#         "afternoon": "",
#         "evening": "Good evening! 🌙 ",
#         "night": "Hey, it's late I hope you're taking care of yourself ",
#     },
#     "ar": {
#         "morning": "صباح الخير! ☀️ ",
#         "afternoon": "",
#         "evening": "مساء الخير! 🌙 ",
#         "night": "الوقت متأخر إن شاء الله  تكون بخير",
#     },
# }

# _CATEGORY_EMOTION = {
#     "greeting": ("joy", 0.95),
#     "gratitude": ("joy", 0.90),
#     "goodbye": ("joy", 0.85),
#     "out_of_scope": ("surprise", 0.70),
# }


# def _quick_response(category: str, lang: str, is_returning: bool = False) -> str:
#     """Build a quick-response string with time-of-day opener and returning-user variant."""
#     pool_key = lang
#     if is_returning and f"{lang}_returning" in _QUICK_RESPONSES.get(category, {}):
#         pool_key = f"{lang}_returning"

#     cat_responses = _QUICK_RESPONSES.get(category, _QUICK_RESPONSES["greeting"])
#     pool = cat_responses.get(pool_key, cat_responses.get(lang, cat_responses["en"]))
#     response = random.choice(pool)

#     if category == "greeting":
#         period = _get_time_period()
#         opener = _TIME_OPENERS.get(lang, _TIME_OPENERS["en"]).get(period, "")
#         if opener and not response.startswith(opener.strip()[:5]):
#             response = opener + response
#     return response


# # ========================================================================
# # INTELLIGENCE HEURISTIC — decides answer vs fallback based on retrieval quality
# # ========================================================================


# def _intelligence_heuristic(
#     query: str, chunks: list, emotion: Optional[str] = None
# ) -> dict:
#     """Evaluates retrieval quality and decides whether to use RAG chunks or fall back."""
#     if not chunks:
#         return {
#             "chunks_relevant": False,
#             "relevant_chunk_indices": [],
#             "rewritten_query": query,
#             "quality_score": 1,
#             "action": "fallback",
#             "reasoning": "No chunks retrieved",
#         }

#     best_similarity = max(c["similarity"] for c in chunks)

#     priority_topics = EMOTION_TOPIC_MAP.get(emotion, [])
#     topic_matches = 0
#     if priority_topics:
#         for c in chunks:
#             for t in c.get("topics", []):
#                 if t in priority_topics:
#                     topic_matches += 1

#     relevant_indices = [i for i, c in enumerate(chunks) if c["similarity"] >= 0.40]
#     if not relevant_indices:
#         relevant_indices = list(range(len(chunks)))

#     if best_similarity >= 0.70:
#         quality = 5
#     elif best_similarity >= 0.55:
#         quality = 4
#     elif best_similarity >= 0.45:
#         quality = 3
#     elif best_similarity >= 0.35:
#         quality = 2
#     else:
#         quality = 1

#     if topic_matches >= 2 and quality < 5:
#         quality += 1

#     if best_similarity >= 0.45 or (best_similarity >= 0.35 and topic_matches >= 1):
#         action = "answer"
#         reasoning = (
#             f"Best similarity {best_similarity:.2f}, {topic_matches} topic matches"
#         )
#     else:
#         action = "fallback"
#         reasoning = (
#             f"Weak similarity {best_similarity:.2f}, insufficient topic relevance"
#         )

#     return {
#         "chunks_relevant": action == "answer",
#         "relevant_chunk_indices": relevant_indices,
#         "rewritten_query": query,
#         "quality_score": quality,
#         "action": action,
#         "reasoning": reasoning,
#     }


# # ========================================================================
# # PROMPT BUILDER — constructs the full therapist system prompt
# # ========================================================================


# def _build_therapist_prompt(
#     query: str,
#     chunks: list,
#     emotion: Optional[str] = None,
#     emotion_conf: Optional[float] = None,
#     language: Optional[str] = None,
#     response_style: Optional[str] = None,
#     crisis_flag: bool = False,
#     prior_crisis: bool = False,
#     country: str = "Unknown",
# ) -> str:
#     sections = [THERAPIST_BASE_PROMPT]

#     if crisis_flag:
#         hotline_info = get_hotline(country)
#         lang_key = "ar" if language == "ar" else "en"
#         sections.append(
#             "⚠ CRISIS CONTEXT ACTIVE\nInclude these resources naturally at the end:\n"
#             + CRISIS_RESOURCES_TEMPLATE[lang_key].format(**hotline_info)
#         )
#     elif prior_crisis:
#         sections.append(
#             "Note: This user previously exhibited crisis signals in this session. Maintain high empathy and monitor for escalation, but do NOT include hotline resources right now unless they ask or escalate again."
#         )

#     if emotion:
#         sections.append(f"Detected emotion: {emotion}")

#     if language == "en":
#         sections.append(
#             "Language Instruction:\n"
#             "- The user spoke in English.\n"
#             "- You MUST respond ENTIRELY and ONLY in English.\n"
#             "- Do NOT include any Arabic words, Arabic characters, or Franco-Arabic transliterations (e.g. do NOT write Arabic phrases using English letters like 'Ana asif').\n"
#             "- Absolutely NO bilingual translation block (do NOT provide the response in both languages)."
#         )
#     elif language == "ar":
#         sections.append(
#             "Language Instruction:\n"
#             "- The user spoke in Arabic.\n"
#             "- You MUST respond ENTIRELY and ONLY in simple, warm Egyptian Arabic (عامية مصرية بسيطة ودافئة ولهجة مصرية عامية حقيقية).\n"
#             "- Do NOT use Modern Standard Arabic (الفصحى).\n"
#             "- Do NOT include any non-Arabic characters, non-Arabic words (no English, no Chinese, no Japanese, no French, no Latin script, and no Franco-Arabic). The response must contain ONLY Arabic script letters and Egyptian phrases.\n"
#             "- Absolutely NO bilingual translation block (do NOT provide the response in both languages)."
#         )

#     if chunks:
#         sections.append(
#             "Clinical Knowledge:\n"
#             + "\n".join([c["response"][:400] for c in chunks[:3]])
#         )

#     return "\n\n".join(sections)


# # ========================================================================
# # NLP PIPELINE CLASS — main orchestration engine
# # ========================================================================


# class NLPPipeline:
#     """
#     Full orchestration pipeline matching the original project logic:
#     - Hardcoded crisis first-line defense
#     - Quick-response fast path (<1ms, zero API calls)
#     - Parallel language + emotion detection via ThreadPoolExecutor
#     - LLM-based intent classification (Gemini)
#     - Out-of-scope & direct routing fast exits
#     - RAG retrieval + emotion reranking + intelligence heuristic
#     - Therapist LLM generation (Gemini) with sliding history
#     - Per-stage timing + JSONL logging
#     """

#     def __init__(self):
#         self._groq_configured = False
#         self._client = None

#     def _ensure_groq(self):
#         if not self._groq_configured:
#             if not GROQ_API_KEY:
#                 raise ValueError("GROQ_API_KEY is not set.")
#             from groq import AsyncGroq

#             self._client = AsyncGroq(api_key=GROQ_API_KEY)
#             self._groq_configured = True

#     async def _call_therapist_llm(
#         self, query: str, prompt: str, history: Optional[list] = None
#     ) -> str:
#         """Calls Groq with system instruction, conversation history, and user query."""
#         self._ensure_groq()

#         messages = [{"role": "system", "content": prompt}]

#         # Convert session history
#         if history:
#             for msg in history[-12:]:
#                 role = "user" if msg["role"] == "user" else "assistant"
#                 messages.append({"role": role, "content": msg["content"]})

#         messages.append({"role": "user", "content": query})

#         try:
#             response = await self._client.chat.completions.create(
#                 model=GROQ_MODEL, messages=messages, temperature=0.75, max_tokens=700
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             logger.error(f"Groq Chat Completion Error: {e}", exc_info=True)
#             return FALLBACK_GENERAL

#     # ================================================================
#     # RUN PIPELINE — main entry point
#     # ================================================================
#     async def run(
#         self, query: str, session: SessionMemory, country: str = "Unknown"
#     ) -> Dict[str, Any]:
#         t_start = time.time()
#         timings = {}
#         prior_crisis = session.prior_crisis
#         history = session.get_history()

#         # ──────────────────────────────────────────────────────
#         # 🚨 FIRST LINE OF DEFENSE: hardcoded crisis signal check
#         # ──────────────────────────────────────────────────────
#         normalized_query = query.lower().strip()
#         has_hardcoded_crisis = intent_classifier.has_crisis_signals(normalized_query)

#         # ──────────────────────────────────────────────────────
#         # Stage 0: QUICK-RESPONSE FAST-PATH (only if no crisis)
#         # ──────────────────────────────────────────────────────
#         if not has_hardcoded_crisis and not prior_crisis:
#             t_quick = time.time()
#             quick_result = _detect_quick_response(query)
#             if quick_result:
#                 category, detected_lang = quick_result

#                 # Out-of-scope immediate exit
#                 if category == "out_of_scope":
#                     answer = _quick_response(category, detected_lang)
#                     timings["quick_response_ms"] = round(
#                         (time.time() - t_quick) * 1000, 1
#                     )
#                     if session:
#                         session.add_turn(
#                             query,
#                             answer,
#                             "surprise",
#                             0.70,
#                             detected_lang,
#                             "out_of_scope",
#                             False,
#                         )
#                     output = {
#                         "answer": answer,
#                         "sources": [],
#                         "emotion": "surprise",
#                         "emotion_conf": 0.70,
#                         "language": detected_lang,
#                         "intent": "out_of_scope",
#                         "routing": "direct",
#                         "crisis_flag": False,
#                         "action_taken": "out_of_scope_fallback",
#                         "quality_score": 5,
#                         "latency_ms": round((time.time() - t_start) * 1000, 1),
#                         "timings": timings,
#                     }
#                     _log_pipeline_interaction(query, output)
#                     return output

#                 # Other quick categories (greeting, gratitude, goodbye)
#                 is_returning = session is not None and session.turn_count > 0
#                 answer = _quick_response(category, detected_lang, is_returning)
#                 emotion, emotion_conf = _CATEGORY_EMOTION.get(category, ("joy", 0.90))
#                 timings["quick_response_ms"] = round((time.time() - t_quick) * 1000, 1)
#                 if session:
#                     session.add_turn(
#                         query,
#                         answer,
#                         emotion,
#                         emotion_conf,
#                         detected_lang,
#                         category,
#                         False,
#                     )
#                 output = {
#                     "answer": answer,
#                     "sources": [],
#                     "emotion": emotion,
#                     "emotion_conf": emotion_conf,
#                     "language": detected_lang,
#                     "intent": category,
#                     "routing": category,
#                     "crisis_flag": False,
#                     "action_taken": category,
#                     "quality_score": 5,
#                     "latency_ms": round((time.time() - t_start) * 1000, 1),
#                     "timings": timings,
#                 }
#                 _log_pipeline_interaction(query, output)
#                 return output

#         # ──────────────────────────────────────────────────────
#         # Stage 1: PARALLEL Language + Emotion Detection
#         # ──────────────────────────────────────────────────────
#         t_parallel = time.time()
#         lang_result, emotion_result = await asyncio.gather(
#             run_in_threadpool(language_detector.detect, query),
#             run_in_threadpool(emotion_classifier.classify, query),
#         )

#         language = lang_result["prediction"]
#         if language not in ("en", "ar"):
#             language = "en"
#         timings["language_ms"] = round((time.time() - t_parallel) * 1000, 1)

#         # Clear history if language changed from last turn to avoid language mixing
#         if session.history:
#             last_msg = session.history[-1]["content"]
#             has_arabic = any("\u0600" <= c <= "\u06ff" for c in last_msg)
#             current_is_english = language == "en"
#             if (current_is_english and has_arabic) or (
#                 not current_is_english and not has_arabic
#             ):
#                 session.history = []
#                 logger.info(
#                     f"Cleared session history due to language switch to {language}."
#                 )

#         emotion = emotion_result["emotion"]
#         emotion_conf = emotion_result["confidence"]
#         timings["emotion_ms"] = round((time.time() - t_parallel) * 1000, 1)

#         # ──────────────────────────────────────────────────────
#         # Stage 2: INTENT CLASSIFICATION (Gemini LLM)
#         # ──────────────────────────────────────────────────────
#         t_intent = time.time()
#         intent_result = await intent_classifier.classify(
#             query, detected_emotion=emotion, detected_language=language
#         )
#         routing = intent_result.get("routing", "rag")
#         extracted_intent = intent_result.get("intent", "asking_mental_health_question")

#         # Crisis flag = hardcoded OR LLM-detected OR emotion risk
#         crisis_flag = (
#             intent_result.get("crisis_flag", False)
#             or has_hardcoded_crisis
#             or emotion_result.get("risk_flag", False)
#         )
#         response_style = (
#             "crisis_intervention"
#             if crisis_flag
#             else intent_result.get("response_style", "empathetic_support")
#         )

#         intent = "asking_mental_health_question" if crisis_flag else extracted_intent
#         timings["intent_ms"] = round((time.time() - t_intent) * 1000, 1)

#         # ──────────────────────────────────────────────────────
#         # Stage 3a: OUT-OF-SCOPE EXIT (post-intent)
#         # ──────────────────────────────────────────────────────
#         if intent == "out_of_scope" or routing == "out_of_scope":
#             answer = _quick_response("out_of_scope", language)
#             if session:
#                 session.add_turn(
#                     query,
#                     answer,
#                     emotion,
#                     emotion_conf,
#                     language,
#                     "out_of_scope",
#                     False,
#                 )
#             output = {
#                 "answer": answer,
#                 "sources": [],
#                 "emotion": emotion,
#                 "emotion_conf": emotion_conf,
#                 "language": language,
#                 "intent": "out_of_scope",
#                 "routing": "direct",
#                 "crisis_flag": False,
#                 "action_taken": "out_of_scope_fallback",
#                 "quality_score": 5,
#                 "latency_ms": round((time.time() - t_start) * 1000, 1),
#                 "timings": timings,
#             }
#             _log_pipeline_interaction(query, output)
#             return output

#         # ──────────────────────────────────────────────────────
#         # Stage 3b: DIRECT ROUTING (no RAG needed)
#         # ──────────────────────────────────────────────────────
#         if routing == "direct" and not (crisis_flag or prior_crisis):
#             t_llm = time.time()
#             prompt = _build_therapist_prompt(
#                 query,
#                 [],
#                 emotion,
#                 emotion_conf,
#                 language,
#                 response_style,
#                 country=country,
#             )
#             answer = await self._call_therapist_llm(query, prompt, history)
#             timings["therapist_ms"] = round((time.time() - t_llm) * 1000, 1)
#             if session:
#                 session.add_turn(
#                     query, answer, emotion, emotion_conf, language, intent, False
#                 )
#             output = {
#                 "answer": answer,
#                 "sources": [],
#                 "emotion": emotion,
#                 "emotion_conf": emotion_conf,
#                 "language": language,
#                 "intent": intent,
#                 "routing": "direct",
#                 "crisis_flag": False,
#                 "action_taken": "direct",
#                 "quality_score": 5,
#                 "latency_ms": round((time.time() - t_start) * 1000, 1),
#                 "timings": timings,
#             }
#             _log_pipeline_interaction(query, output)
#             return output

#         # ──────────────────────────────────────────────────────
#         # Stage 4: RAG RETRIEVAL + EMOTION RERANKING
#         # ──────────────────────────────────────────────────────
#         t_retrieve = time.time()
#         chunks = await rag_service.retrieve_and_rerank(query, emotion=emotion)
#         timings["retrieval_ms"] = round((time.time() - t_retrieve) * 1000, 1)

#         # ──────────────────────────────────────────────────────
#         # Stage 5: INTELLIGENCE HEURISTIC
#         # ──────────────────────────────────────────────────────
#         t_intel = time.time()
#         if crisis_flag:
#             action = "crisis"
#             final_chunks = chunks
#             intel = {"quality_score": 5, "reasoning": "Crisis forced"}
#         else:
#             intel = _intelligence_heuristic(query, chunks, emotion)
#             action = intel.get("action", "answer")

#             if action == "fallback" and chunks:
#                 final_chunks = chunks
#                 action = "answer"
#             elif action == "fallback":
#                 final_chunks = []
#             else:
#                 final_chunks = chunks
#         timings["intelligence_ms"] = round((time.time() - t_intel) * 1000, 1)

#         # ──────────────────────────────────────────────────────
#         # Stage 6: THERAPIST LLM GENERATION (Gemini)
#         # ──────────────────────────────────────────────────────
#         t_llm = time.time()
#         prompt = _build_therapist_prompt(
#             query,
#             final_chunks,
#             emotion,
#             emotion_conf,
#             language,
#             response_style,
#             crisis_flag or (action == "crisis"),
#             prior_crisis,
#             country,
#         )
#         answer = await self._call_therapist_llm(query, prompt, history)
#         timings["therapist_ms"] = round((time.time() - t_llm) * 1000, 1)

#         # Build sources list for API response
#         sources = (
#             [
#                 {
#                     "excerpt": c["context"][:80] + "...",
#                     "similarity": c["similarity"],
#                     "topics": c["topics"],
#                     "risk_level": c["risk_level"],
#                 }
#                 for c in final_chunks
#             ]
#             if final_chunks
#             else []
#         )

#         # Update session memory
#         if session:
#             session.add_turn(
#                 query,
#                 answer,
#                 emotion,
#                 emotion_conf,
#                 language,
#                 intent,
#                 crisis_flag,
#                 topics=[t for c in final_chunks for t in c.get("topics", [])],
#             )

#         try:
#             quality_score = int(intel.get("quality_score", 3))
#         except (ValueError, TypeError):
#             quality_score = 3

#         output = {
#             "answer": answer,
#             "sources": sources,
#             "emotion": emotion,
#             "emotion_conf": emotion_conf,
#             "language": language,
#             "intent": intent,
#             "routing": "rag",
#             "crisis_flag": crisis_flag,
#             "action_taken": action,
#             "quality_score": quality_score,
#             "latency_ms": round((time.time() - t_start) * 1000, 1),
#             "timings": timings,
#         }
#         _log_pipeline_interaction(query, output)
#         return output


# # Global pipeline instance
# nlp_pipeline = NLPPipeline()

import re
import json
import time
import random
import logging
import asyncio
from collections import Counter
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from fastapi.concurrency import run_in_threadpool
from app.config import GROQ_API_KEY, GROQ_MODEL, BASE_DIR
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
                "topics": src.get("topics", []),
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


# THREAD POOL removed we now use fastapi.concurrency.run_in_threadpool
# ========================================================================
# THERAPIST SYSTEM PROMPT — matches original (3-5 paragraphs)
# ========================================================================
THERAPIST_BASE_PROMPT = """
You are a warm, deeply empathetic, and highly conversational mental health supporter.
Your core principles:
1. BE NATURAL AND CONVERSATIONAL: Talk like a caring human, not a robotic textbook therapist. Do not awkwardly repeat the user's exact words back to them.
2. MATCH THE TONE: If the user is casual, lighthearted, or asking a simple question, be friendly, brief, and conversational. Save deep therapeutic reflection ONLY for actual distress or emotional pain.
3. NEVER MINIMIZE PAIN: If they share pain, validate it warmly before offering any advice.
4. RESPONSE LENGTH — STRICT RULES:
    - Simple greeting or casual message → 1 short paragraph (2–3 sentences max).
    - Mild sadness or venting → 1–2 paragraphs.
    - Serious distress, grief, or trauma → 2–3 focused paragraphs. Do NOT pad with filler.
    - NEVER write more than 3 paragraphs regardless of the topic.
    - Quality over quantity: one genuine, direct sentence beats three hollow ones.
5. EMOJI RULES — STRICT:
    - Use a MAXIMUM of 2 emojis per response. Often 1 is enough. Zero is fine for serious topics.
    - Do NOT use emojis to decorate every sentence. They dilute emotional impact.
    - In a crisis or serious distress context: 0–1 emojis only.
    - Acceptable emojis for warmth: 💙 🤍 🌷 🫂 — avoid cartoon/food/party emojis entirely.
6. GENTLE GUIDANCE: You may ask one natural question to keep the conversation flowing, but only if it makes sense. Do not force a probing psychological question on a casual topic.
7. LANGUAGE & CULTURAL STYLE:
    - Always respond STRICTLY in the single language the user used.
    - If the user writes in English, respond ENTIRELY in English. Do NOT include any Arabic words, Arabic characters, or Franco-Arabic transliterations.
    - If the user writes in Arabic, respond ENTIRELY in simple, warm Egyptian Arabic (عامية مصرية بسيطة ولهجة مصرية عامية دافئة وحقيقية). Do NOT use Modern Standard Arabic (الفصحى) except for extremely common phrases.
    - NEVER mix languages or provide translations or transliterations in your response.
8. CRISIS HANDLING: If a crisis is detected, prioritize immediate safety, extreme warmth, and direct the user to the provided resources without being pushy.
"""

FALLBACK_GENERAL = (
    "I am having a little trouble reaching my full resources right now, "
    "but I am here and I am listening. Can you tell me a little more about what brought you here today?"
)


# ========================================================================
# EMOTION → TOPIC MAP (for intelligence heuristic & reranking)
# ========================================================================
EMOTION_TOPIC_MAP = {
    "sadness": ["depression", "grief_loss", "self_esteem", "suicidal", "loneliness"],
    "fear": ["anxiety", "trauma_ptsd", "stress", "sleep"],
    "anger": ["anger", "relationships", "stress"],
    "love": ["relationships", "self_esteem"],
    "joy": [],
    "surprise": [],
    "uncertain": [],
}


# ========================================================================
# QUICK-RESPONSE SYSTEM — Fast path, zero API calls, <1ms
# ========================================================================


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/emoji, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(
        r"[^\w\s\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u9FFF\u0400-\u04FF]",
        "",
        text,
    )
    return " ".join(text.split())


_QUICK_PATTERNS: dict[str, dict[str, list[str]]] = {
    "greeting": {
        "en": [
            "hi",
            "hello",
            "hey",
            "heyy",
            "heyyy",
            "howdy",
            "yo",
            "good morning",
            "good afternoon",
            "good evening",
            "good night",
            "morning",
            "evening",
            "greetings",
            "whats up",
            "sup",
            "hows it going",
            "hi there",
            "hello there",
            "hey there",
            "how are you",
            "how r u",
            "how are u",
        ],
        "ar": [
            "مرحبا",
            "مرحبه",
            "اهلا",
            "أهلا",
            "ازيك",
            "هلا",
            "هلا والله",
            "السلام عليكم",
            "سلام عليكم",
            "سلام",
            "صباح الخير",
            "مساء الخير",
            "صباح النور",
            "مساء النور",
            "كيف حالك",
            "كيفك",
            "شلونك",
            "كيف الحال",
            "اهلا وسهلا",
            "أهلا وسهلا",
            "يا هلا",
            "هاي",
            "هالو",
        ],
    },
    "gratitude": {
        "en": [
            "thank you",
            "thanks",
            "thank u",
            "thx",
            "ty",
            "thanks a lot",
            "thank you so much",
            "thanks so much",
            "much appreciated",
            "appreciate it",
        ],
        "ar": [
            "شكرا",
            "شكراً",
            "شكرا لك",
            "شكراً لك",
            "مشكور",
            "مشكورة",
            "الله يعطيك العافية",
            "يعطيك العافية",
            "جزاك الله خيرا",
            "تسلم",
            "تسلمي",
        ],
    },
    "goodbye": {
        "en": [
            "bye",
            "goodbye",
            "good bye",
            "see you",
            "see ya",
            "take care",
            "bye bye",
        ],
        "ar": ["مع السلامة", "باي", "في أمان الله", "الله يحفظك", "إلى اللقاء", "سلام"],
    },
    "out_of_scope": {
        "en": [
            "whats the weather",
            "tell me a joke",
            "what time is it",
            "write code",
            "how to code",
            "weather today",
            "recipe for",
            "who won the game",
            "sing a song",
            "make me a script",
            "generate code",
            "what is the capital of",
            "capital of",
        ],
        "ar": [
            "كيف الطقس",
            "كم الساعة",
            "احكيلي نكتة",
            "مين انت",
            "شو اسمك",
            "اكتب كود",
            "برمجة",
            "طريقة عمل",
            "اخبار الرياضة",
            "قول نكتة",
        ],
    },
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

# ========================================================================
# OUT-OF-SCOPE KEYWORD GUARD
# Catches off-topic requests even when emotionally framed, e.g.
# "I'm so sad, how do I make pizza?" — before the LLM ever sees them.
# Add keywords here to block new topics without touching intent classifier.
# ========================================================================
_OOS_KEYWORDS_EN: set[str] = {
    # Food & cooking
    "pizza",
    "recipe",
    "cook",
    "cooking",
    "bake",
    "baking",
    "ingredient",
    "ingredients",
    "flour",
    "dough",
    "yeast",
    "oven",
    "meal",
    "food",
    "restaurant",
    "cafe",
    "breakfast",
    "lunch",
    "dinner",
    "dessert",
    "sandwich",
    "pasta",
    "burger",
    "sushi",
    "chicken breast",
    "how to make",
    # Weather
    "weather",
    "forecast",
    "temperature",
    "rain",
    "snow",
    "sunny",
    # News / sports
    "news",
    "sport",
    "sports",
    "game",
    "match",
    "score",
    "team",
    "football",
    "basketball",
    "soccer",
    "cricket",
    "f1",
    "formula 1",
    # Tech / code
    "code",
    "coding",
    "programming",
    "script",
    "html",
    "python",
    "javascript",
    "write a function",
    "debug",
    "api",
    "database",
    # Other
    "joke",
    "jokes",
    "riddle",
    "song lyrics",
    "translate",
    "capital of",
    "how tall is",
    "who invented",
    "when was",
}

_OOS_KEYWORDS_AR: set[str] = {
    # Food
    "بيتزا",
    "طريقة عمل",
    "وصفة",
    "وصفات",
    "طبخ",
    "اكل",
    "أكل",
    "مطبخ",
    "طحين",
    "عجين",
    "فرن",
    "مطعم",
    "فطور",
    "غداء",
    "عشاء",
    "حلويات",
    "كيكة",
    "برجر",
    "شاورما",
    "مكرونة",
    # Weather / general
    "الطقس",
    "طقس",
    "درجة الحرارة",
    "امطار",
    "شمس",
    # Sports / news
    "اخبار",
    "رياضة",
    "كرة قدم",
    "مباراة",
    "نتيجة",
    "فريق",
    # Tech
    "كود",
    "برمجة",
    "اكتب كود",
    "برنامج",
    # Other
    "نكتة",
    "نكت",
    "اغنية",
    "ترجمة",
    "كلمات اغنية",
}


def _is_out_of_scope_by_keywords(text: str, language: str) -> bool:
    """
    Keyword-based OOS guard that fires BEFORE the LLM intent call.
    Intentionally broad — catches emotionally-framed off-topic requests
    like 'I'm sad, make me a pizza recipe'.
    Returns True if the query is clearly off-topic.
    """
    lower = text.lower()
    if language == "ar":
        return any(kw in lower for kw in _OOS_KEYWORDS_AR)
    # Default to English keyword set (also covers mixed/unknown)
    return any(kw in lower for kw in _OOS_KEYWORDS_EN)


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
                remainder = remainder[len(filler) :].lstrip()
                break
        if not remainder:
            break

        matched = False
        for pat, lang, cat in _QUICK_ALL:
            if remainder == pat or remainder.startswith(pat + " "):
                detected.append((cat, lang))
                remainder = remainder[len(pat) :].lstrip()
                matched = True
                break
        if not matched:
            return None

    if not remainder and detected:
        cats = [c for c, _ in detected]
        langs = [lang for _, lang in detected]
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
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


_QUICK_RESPONSES: dict[str, dict[str, list[str]]] = {
    "greeting": {
        "en": [
            "Hello! 😊 I'm really glad you're here this is a safe space what's on your mind today?"
        ],
        "ar": [
            "أهلًا بيك! 😊 مجرد إنك قررت تتكلم خطوة مهمة وشجاعة احكي براحتك وأنا هسمعك من غير أي حكم أو ضغط 🤗"
        ],
        "ar_returning": [
            "أهلًا بيك من جديد 😊 سعيد إني بشوفك تاني إيه الأخبار من آخر مرة اتكلمنا؟ 💙"
        ],
    },
    "gratitude": {
        "en": [
            "You're so welcome! 💛 Remember I'm always here whenever you need to talk"
        ],
        "ar": [
            "العفو 😊 إنت أظهرت قوة حقيقية بإنك انفتحت وحكيت اعتني بنفسك ولا تتردد إنك ترجع في أي وقت. 💛"
        ],
    },
    "goodbye": {
        "en": ["Take care of yourself 💛 You're not alone in this."],
        "ar": [
            "اعتني بنفسك 💛وتذكر انا هنا وقت ما تحتاج تتكلم في أي وقت ما إنت مش لوحدك مع السلامة! 😊"
        ],
    },
    "out_of_scope": {
        "en": [
            "I wish I could help with that! 😊 My expertise is specifically in mental health support."
        ],
        "ar": [
            "أقدر فضولك! 😊 أنا متخصص في دعم الصحة النفسية والعاطفية فقط وما قدرش أساعدك فى هذا الموضوع بس لو شايل هم في قلبك أنا هسمعك. 💛"
        ],
    },
}

_TIME_OPENERS = {
    "en": {
        "morning": "Good morning! ☀️ ",
        "afternoon": "",
        "evening": "Good evening! 🌙 ",
        "night": "Hey, it's late I hope you're taking care of yourself ",
    },
    "ar": {
        "morning": "صباح الخير! ☀️ ",
        "afternoon": "",
        "evening": "مساء الخير! 🌙 ",
        "night": "الوقت متأخر إن شاء الله  تكون بخير",
    },
}

_CATEGORY_EMOTION = {
    "greeting": ("joy", 0.95),
    "gratitude": ("joy", 0.90),
    "goodbye": ("joy", 0.85),
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


def _intelligence_heuristic(
    query: str, chunks: list, emotion: Optional[str] = None
) -> dict:
    """Evaluates retrieval quality and decides whether to use RAG chunks or fall back."""
    if not chunks:
        return {
            "chunks_relevant": False,
            "relevant_chunk_indices": [],
            "rewritten_query": query,
            "quality_score": 1,
            "action": "fallback",
            "reasoning": "No chunks retrieved",
        }

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

    if best_similarity >= 0.70:
        quality = 5
    elif best_similarity >= 0.55:
        quality = 4
    elif best_similarity >= 0.45:
        quality = 3
    elif best_similarity >= 0.35:
        quality = 2
    else:
        quality = 1

    if topic_matches >= 2 and quality < 5:
        quality += 1

    if best_similarity >= 0.45 or (best_similarity >= 0.35 and topic_matches >= 1):
        action = "answer"
        reasoning = (
            f"Best similarity {best_similarity:.2f}, {topic_matches} topic matches"
        )
    else:
        action = "fallback"
        reasoning = (
            f"Weak similarity {best_similarity:.2f}, insufficient topic relevance"
        )

    return {
        "chunks_relevant": action == "answer",
        "relevant_chunk_indices": relevant_indices,
        "rewritten_query": query,
        "quality_score": quality,
        "action": action,
        "reasoning": reasoning,
    }


# ========================================================================
# PROMPT BUILDER — constructs the full therapist system prompt
# ========================================================================


def _build_therapist_prompt(
    query: str,
    chunks: list,
    emotion: Optional[str] = None,
    emotion_conf: Optional[float] = None,
    language: Optional[str] = None,
    response_style: Optional[str] = None,
    crisis_flag: bool = False,
    prior_crisis: bool = False,
    country: str = "Unknown",
) -> str:
    sections = [THERAPIST_BASE_PROMPT]

    if crisis_flag:
        hotline_info = get_hotline(country)
        lang_key = "ar" if language == "ar" else "en"
        sections.append(
            "⚠ CRISIS CONTEXT ACTIVE\nInclude these resources naturally at the end:\n"
            + CRISIS_RESOURCES_TEMPLATE[lang_key].format(**hotline_info)
        )
    elif prior_crisis:
        sections.append(
            "Note: This user previously exhibited crisis signals in this session. Maintain high empathy and monitor for escalation, but do NOT include hotline resources right now unless they ask or escalate again."
        )

    if emotion:
        sections.append(f"Detected emotion: {emotion}")

    if language == "en":
        sections.append(
            "Language Instruction:\n"
            "- The user spoke in English.\n"
            "- You MUST respond ENTIRELY and ONLY in English.\n"
            "- Do NOT include any Arabic words, Arabic characters, or Franco-Arabic transliterations (e.g. do NOT write Arabic phrases using English letters like 'Ana asif').\n"
            "- Absolutely NO bilingual translation block (do NOT provide the response in both languages)."
        )
    elif language == "ar":
        sections.append(
            "Language Instruction:\n"
            "- The user spoke in Arabic.\n"
            "- You MUST respond ENTIRELY and ONLY in warm, natural spoken Egyptian Arabic (عامية مصرية دافئة وبسيطة).\n"
            "- Do NOT use Modern Standard Arabic (الفصحى).\n"
            "- Sound like a caring, close friend — NOT a textbook or formal counselor.\n"
            "- Example tone: 'أنا سامعك، وده طبيعي تحس كده — إيه اللي بيضايقك أكتر؟'\n"
            "- Use ONLY Arabic script. Do NOT mix English, Latin alphabet, or any other languages/scripts into your response.\n"
            "- Absolutely NO bilingual translation block (do NOT provide the response in both languages)."
        )

    if chunks:
        if language == "ar":
            sections.append(
                "Clinical Knowledge (reference only — this is in English. "
                "Do NOT translate it literally or mirror its wording. "
                "Use it as background guidance while crafting a natural, warm Arabic response):\n"
                + "\n".join([c["response"][:400] for c in chunks[:3]])
            )
        else:
            sections.append(
                "Clinical Knowledge:\n"
                + "\n".join([c["response"][:400] for c in chunks[:3]])
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
        self._groq_configured = False
        self._client = None

    def _ensure_groq(self):
        if not self._groq_configured:
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set.")
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=GROQ_API_KEY)
            self._groq_configured = True

    async def _call_therapist_llm(
        self, query: str, prompt: str, history: Optional[list] = None
    ) -> str:
        """Calls Groq with system instruction, conversation history, and user query."""
        self._ensure_groq()

        messages = [{"role": "system", "content": prompt}]

        # Convert session history
        if history:
            for msg in history[-12:]:
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})

        messages.append({"role": "user", "content": query})

        try:
            response = await self._client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, temperature=0.4, max_tokens=700
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq Chat Completion Error: {e}", exc_info=True)
            return FALLBACK_GENERAL

    # ================================================================
    # RUN PIPELINE — main entry point
    # ================================================================
    async def run(
        self, query: str, session: SessionMemory, country: str = "Unknown"
    ) -> Dict[str, Any]:
        t_start = time.time()
        timings = {}
        prior_crisis = session.prior_crisis
        history = session.get_history()

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
                    timings["quick_response_ms"] = round(
                        (time.time() - t_quick) * 1000, 1
                    )
                    if session:
                        session.add_turn(
                            query,
                            answer,
                            "surprise",
                            0.70,
                            detected_lang,
                            "out_of_scope",
                            False,
                        )
                    output = {
                        "answer": answer,
                        "sources": [],
                        "emotion": "surprise",
                        "emotion_conf": 0.70,
                        "language": detected_lang,
                        "intent": "out_of_scope",
                        "routing": "direct",
                        "crisis_flag": False,
                        "action_taken": "out_of_scope_fallback",
                        "quality_score": 5,
                        "latency_ms": round((time.time() - t_start) * 1000, 1),
                        "timings": timings,
                    }
                    _log_pipeline_interaction(query, output)
                    return output

                # Other quick categories (greeting, gratitude, goodbye)
                is_returning = session is not None and session.turn_count > 0
                answer = _quick_response(category, detected_lang, is_returning)
                emotion, emotion_conf = _CATEGORY_EMOTION.get(category, ("joy", 0.90))
                timings["quick_response_ms"] = round((time.time() - t_quick) * 1000, 1)
                if session:
                    session.add_turn(
                        query,
                        answer,
                        emotion,
                        emotion_conf,
                        detected_lang,
                        category,
                        False,
                    )
                output = {
                    "answer": answer,
                    "sources": [],
                    "emotion": emotion,
                    "emotion_conf": emotion_conf,
                    "language": detected_lang,
                    "intent": category,
                    "routing": category,
                    "crisis_flag": False,
                    "action_taken": category,
                    "quality_score": 5,
                    "latency_ms": round((time.time() - t_start) * 1000, 1),
                    "timings": timings,
                }
                _log_pipeline_interaction(query, output)
                return output

        # ──────────────────────────────────────────────────────
        # Stage 1: PARALLEL Language + Emotion Detection
        # ──────────────────────────────────────────────────────
        t_parallel = time.time()
        lang_result, emotion_result = await asyncio.gather(
            run_in_threadpool(language_detector.detect, query),
            run_in_threadpool(emotion_classifier.classify, query),
        )

        language = lang_result["prediction"]
        # Unicode-based override: short/ambiguous texts may be mis-detected.
        # If the detector is not confident OR returned an unsupported language,
        # fall back to script inspection — Arabic Unicode block wins.
        if not lang_result.get("trusted", True) or language not in ("en", "ar"):
            has_arabic_chars = any("\u0600" <= c <= "\u06ff" for c in query)
            language = "ar" if has_arabic_chars else "en"
        timings["language_ms"] = round((time.time() - t_parallel) * 1000, 1)

        # Clear history if language changed from last turn to avoid language mixing.
        # Compare against the last USER message (not assistant), so Arabic replies
        # from the bot don't falsely trigger a reset when the user writes English next.
        if session.history:
            last_user_msgs = [m for m in session.history if m["role"] == "user"]
            if last_user_msgs:
                last_content = last_user_msgs[-1]["content"]
                has_arabic = any("\u0600" <= c <= "\u06ff" for c in last_content)
                current_is_arabic = language == "ar"
                if has_arabic != current_is_arabic:
                    session.history = []
                    logger.info(
                        f"Cleared session history due to language switch to {language}."
                    )

        emotion = emotion_result["emotion"]
        emotion_conf = emotion_result["confidence"]
        timings["emotion_ms"] = round((time.time() - t_parallel) * 1000, 1)

        # ──────────────────────────────────────────────────────
        # Stage 1b: KEYWORD OUT-OF-SCOPE GUARD (pre-LLM)
        # Catches off-topic requests even when emotionally framed
        # (e.g. "I'm so sad, how do I make pizza?"). This fires
        # before the intent LLM to save latency and prevent the
        # model from being talked into answering OOS questions.
        # ──────────────────────────────────────────────────────
        if not has_hardcoded_crisis and not prior_crisis:
            if _is_out_of_scope_by_keywords(query, language):
                answer = _quick_response("out_of_scope", language)
                if session:
                    session.add_turn(
                        query,
                        answer,
                        emotion,
                        emotion_conf,
                        language,
                        "out_of_scope",
                        False,
                    )
                output = {
                    "answer": answer,
                    "sources": [],
                    "emotion": emotion,
                    "emotion_conf": emotion_conf,
                    "language": language,
                    "intent": "out_of_scope",
                    "routing": "direct",
                    "crisis_flag": False,
                    "action_taken": "out_of_scope_keyword_guard",
                    "quality_score": 5,
                    "latency_ms": round((time.time() - t_start) * 1000, 1),
                    "timings": timings,
                }
                _log_pipeline_interaction(query, output)
                return output

        # ──────────────────────────────────────────────────────
        # Stage 2: INTENT CLASSIFICATION (Gemini LLM)
        # ──────────────────────────────────────────────────────
        t_intent = time.time()
        intent_result = await intent_classifier.classify(
            query, detected_emotion=emotion, detected_language=language
        )
        routing = intent_result.get("routing", "rag")
        extracted_intent = intent_result.get("intent", "asking_mental_health_question")

        # Crisis flag = hardcoded OR LLM-detected OR emotion risk
        crisis_flag = (
            intent_result.get("crisis_flag", False)
            or has_hardcoded_crisis
            or emotion_result.get("risk_flag", False)
        )
        response_style = (
            "crisis_intervention"
            if crisis_flag
            else intent_result.get("response_style", "empathetic_support")
        )

        intent = "asking_mental_health_question" if crisis_flag else extracted_intent
        timings["intent_ms"] = round((time.time() - t_intent) * 1000, 1)

        # ──────────────────────────────────────────────────────
        # Stage 3a: OUT-OF-SCOPE EXIT (post-intent)
        # ──────────────────────────────────────────────────────
        if intent == "out_of_scope" or routing == "out_of_scope":
            answer = _quick_response("out_of_scope", language)
            if session:
                session.add_turn(
                    query,
                    answer,
                    emotion,
                    emotion_conf,
                    language,
                    "out_of_scope",
                    False,
                )
            output = {
                "answer": answer,
                "sources": [],
                "emotion": emotion,
                "emotion_conf": emotion_conf,
                "language": language,
                "intent": "out_of_scope",
                "routing": "direct",
                "crisis_flag": False,
                "action_taken": "out_of_scope_fallback",
                "quality_score": 5,
                "latency_ms": round((time.time() - t_start) * 1000, 1),
                "timings": timings,
            }
            _log_pipeline_interaction(query, output)
            return output

        # ──────────────────────────────────────────────────────
        # Stage 3b: DIRECT ROUTING (no RAG needed)
        # ──────────────────────────────────────────────────────
        if routing == "direct" and not (crisis_flag or prior_crisis):
            t_llm = time.time()
            prompt = _build_therapist_prompt(
                query,
                [],
                emotion,
                emotion_conf,
                language,
                response_style,
                country=country,
            )
            answer = await self._call_therapist_llm(query, prompt, history)
            timings["therapist_ms"] = round((time.time() - t_llm) * 1000, 1)
            if session:
                session.add_turn(
                    query, answer, emotion, emotion_conf, language, intent, False
                )
            output = {
                "answer": answer,
                "sources": [],
                "emotion": emotion,
                "emotion_conf": emotion_conf,
                "language": language,
                "intent": intent,
                "routing": "direct",
                "crisis_flag": False,
                "action_taken": "direct",
                "quality_score": 5,
                "latency_ms": round((time.time() - t_start) * 1000, 1),
                "timings": timings,
            }
            _log_pipeline_interaction(query, output)
            return output

        # ──────────────────────────────────────────────────────
        # Stage 4: RAG RETRIEVAL + EMOTION RERANKING
        # ──────────────────────────────────────────────────────
        t_retrieve = time.time()
        chunks = await rag_service.retrieve_and_rerank(query, emotion=emotion)
        timings["retrieval_ms"] = round((time.time() - t_retrieve) * 1000, 1)

        # ──────────────────────────────────────────────────────
        # Stage 5: INTELLIGENCE HEURISTIC
        # ──────────────────────────────────────────────────────
        t_intel = time.time()
        if crisis_flag:
            action = "crisis"
            final_chunks = chunks
            intel = {"quality_score": 5, "reasoning": "Crisis forced"}
        else:
            intel = _intelligence_heuristic(query, chunks, emotion)
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
            query,
            final_chunks,
            emotion,
            emotion_conf,
            language,
            response_style,
            crisis_flag or (action == "crisis"),
            prior_crisis,
            country,
        )
        answer = await self._call_therapist_llm(query, prompt, history)
        timings["therapist_ms"] = round((time.time() - t_llm) * 1000, 1)

        # Build sources list for API response
        sources = (
            [
                {
                    "excerpt": c["context"][:80] + "...",
                    "similarity": c["similarity"],
                    "topics": c["topics"],
                    "risk_level": c["risk_level"],
                }
                for c in final_chunks
            ]
            if final_chunks
            else []
        )

        # Update session memory
        if session:
            session.add_turn(
                query,
                answer,
                emotion,
                emotion_conf,
                language,
                intent,
                crisis_flag,
                topics=[t for c in final_chunks for t in c.get("topics", [])],
            )

        try:
            quality_score = int(intel.get("quality_score", 3))
        except (ValueError, TypeError):
            quality_score = 3

        output = {
            "answer": answer,
            "sources": sources,
            "emotion": emotion,
            "emotion_conf": emotion_conf,
            "language": language,
            "intent": intent,
            "routing": "rag",
            "crisis_flag": crisis_flag,
            "action_taken": action,
            "quality_score": quality_score,
            "latency_ms": round((time.time() - t_start) * 1000, 1),
            "timings": timings,
        }
        _log_pipeline_interaction(query, output)
        return output


# Global pipeline instance
nlp_pipeline = NLPPipeline()
