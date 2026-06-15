from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Feedback
from app.schemas import FeedbackRequest
from app.utils.logging import app_logger
import app.utils.metrics as metrics

router = APIRouter()


@router.post(
    "/feedback", status_code=status.HTTP_201_CREATED, summary="Log user feedback"
)
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Receives user feedback (thumbs up/down) and saves it to the SQLite database."""
    vote = req.vote.strip().lower()
    if vote not in ("up", "down"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vote must be either 'up' or 'down'",
        )

    try:
        feedback = Feedback(
            vote=vote, user_message=req.user_message, bot_response=req.bot_response
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)

        # Increment feedback counter metric with the vote attribute
        metrics.feedback_counter.add(1, {"vote": vote})

        app_logger.info(f"Feedback logged successfully: vote={vote}")
        return {"status": "success", "message": "Feedback saved successfully"}

    except Exception as e:
        db.rollback()
        app_logger.error(f"Failed to save feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save feedback to database",
        )
