from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import List, Dict, Any, Optional
from app.config import QDRANT_URL, QDRANT_API_KEY

COLLECTION_NAME = "mental_health_counseling"
SIMILARITY_GATE = 0.35

EMOTION_TOPIC_MAP = {
    "sadness": ["depression", "grief_loss", "self_esteem", "suicidal", "loneliness"],
    "fear": ["anxiety", "trauma_ptsd", "stress", "sleep"],
    "anger": ["anger", "relationships", "stress"],
    "love": ["relationships", "self_esteem"],
    "joy": [],
    "surprise": [],
    "uncertain": [],
}


class RAGService:
    """Lazy loader and wrapper for Qdrant vector search and emotion-based context reranking."""

    def __init__(self):
        self._embed_model = None
        self._qdrant_client = None
        self._is_loaded = False

    def _load(self):
        if not self._is_loaded:
            self._embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            if not QDRANT_URL or not QDRANT_API_KEY:
                raise ValueError(
                    "Qdrant URL or API Key is missing in environment variables."
                )
            self._qdrant_client = QdrantClient(
                url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60
            )
            self._is_loaded = True

    def _embed(self, text: str) -> List[float]:
        self._load()
        return self._embed_model.encode(text, normalize_embeddings=True).tolist()

    def _adaptive_top_k(self, query: str) -> int:
        words = len(query.split())
        if words <= 8:
            return 3
        if words <= 20:
            return 5
        return 7

    def retrieve_and_rerank(
        self, query: str, emotion: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self._load()
        top_k = self._adaptive_top_k(query)

        try:
            vector = self._embed(query)
            results = self._qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=top_k,
                with_payload=True,
                score_threshold=SIMILARITY_GATE,
            ).points

            chunks = []
            for r in results:
                payload = r.payload or {}
                chunks.append(
                    {
                        "context": payload.get("context", ""),
                        "response": payload.get("response", ""),
                        "topics": payload.get("topics", []),
                        "risk_level": payload.get("risk_level", "low"),
                        "quality_score": payload.get("quality_score", 1),
                        "has_empathy": payload.get("has_empathy", False),
                        "similarity": round(float(r.score), 4),
                    }
                )

            # Emotion Reranking heuristic
            if emotion and chunks:
                priority_topics = EMOTION_TOPIC_MAP.get(emotion, [])
                if priority_topics:
                    # Rerank by similarity score + boost for matching topics + boost for empathy
                    chunks = sorted(
                        chunks,
                        key=lambda c: (
                            c["similarity"]
                            + sum(
                                0.08
                                for t in c.get("topics", [])
                                if t in priority_topics
                            )
                            + (0.05 if c.get("has_empathy") else 0)
                        ),
                        reverse=True,
                    )
            return chunks

        except Exception as e:
            # Return empty if Qdrant fails, allowing system to fallback gracefully
            import logging

            logging.getLogger("app_logger").error(
                f"Qdrant Retrieval Failed: {e}", exc_info=True
            )
            return []


# Global single instance
rag_service = RAGService()
