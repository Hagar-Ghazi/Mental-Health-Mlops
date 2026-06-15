from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import ChatSession, Message
from app.schemas import ChatRequest, ChatResponse
from app.services.nlp_pipeline import nlp_pipeline
from app.services.session import session_store
from app.services.crisis import log_crisis_event
from app.utils.logging import app_logger
import app.utils.metrics as metrics

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Send chat message")
def chat(req: ChatRequest, request: Request, db: Session = Depends(get_db)):
    """Accepts a chat message, runs the NLP pipeline (classification + RAG + LLM),

    saves the message to the database, and registers monitoring metrics.
    """
    message_text = req.message.strip()
    if not message_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty"
        )

    # 1. Resolve client IP and country for session tracking
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    country = request.headers.get("cf-ipcountry", "United States")

    # 2. Get or create session memory from memory store
    session = session_store.get_or_create(client_ip)

    # 3. Ensure ChatSession row exists in database
    db_session = (
        db.query(ChatSession).filter(ChatSession.session_id == client_ip).first()
    )
    if not db_session:
        db_session = ChatSession(session_id=client_ip)
        db.add(db_session)
        db.commit()
        db.refresh(db_session)

    # 4. Run the chatbot pipeline
    result = nlp_pipeline.run(query=message_text, session=session, country=country)

    try:
        # 5. Persist messages to the database
        db_user_msg = Message(
            session_id=client_ip,
            role="user",
            content=message_text,
            emotion=result["emotion"],
            emotion_conf=result["emotion_conf"],
            language=result["language"],
            intent=result["intent"],
            crisis_flag=result["crisis_flag"],
        )
        db_bot_msg = Message(
            session_id=client_ip,
            role="assistant",
            content=result["answer"],
            crisis_flag=result["crisis_flag"],
        )
        db.add(db_user_msg)
        db.add(db_bot_msg)

        # Increment turn count
        db_session.turn_count += 1

        # Log crisis event if safety limits breached
        if result["crisis_flag"]:
            db_session.prior_crisis = True
            log_crisis_event(db, client_ip, message_text)

        db.commit()

    except Exception as db_err:
        db.rollback()
        app_logger.error(
            f"Database write failed during /chat transaction: {db_err}", exc_info=True
        )

    # 6. Record telemetry metrics via OpenTelemetry SDK
    metrics.intent_counter.add(1, {"intent": result["intent"]})
    metrics.emotion_counter.add(1, {"emotion": result.get("emotion") or "uncertain"})
    metrics.latency_histogram.record(result["latency_ms"], {"intent": result["intent"]})
    metrics.msg_length_histogram.record(len(message_text))

    # Record similarity scores of retrieved chunks if any
    for score in result.get("rag_scores", []):
        metrics.rag_scores_histogram.record(score)

    return ChatResponse(
        response=result["answer"],
        answer=result["answer"],
        session_id=client_ip,
        emotion=result["emotion"],
        emotion_conf=result["emotion_conf"],
        language=result["language"],
        intent=result["intent"],
        crisis_flag=result["crisis_flag"],
    )
