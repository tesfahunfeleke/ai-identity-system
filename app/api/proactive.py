from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.proactive import ProactiveEngine
from app.services.scheduler import SchedulerService

router = APIRouter(prefix="/proactive", tags=["proactive"])
proactive_engine = ProactiveEngine()
scheduler = SchedulerService()


class TextAnalysisRequest(BaseModel):
    text: str


@router.post("/detect-habits")
async def detect_habits(request: TextAnalysisRequest):
    """Detect habits from text."""
    try:
        habits = proactive_engine.detect_habits(request.text)
        return {"habits": habits, "count": len(habits)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/detect-goals")
async def detect_goals(request: TextAnalysisRequest):
    """Detect goals from text."""
    try:
        goals = proactive_engine.detect_goals(request.text)
        return {"goals": goals, "count": len(goals)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/insights")
async def get_insights():
    """Get comprehensive insights about the user."""
    try:
        return proactive_engine.get_insights()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/daily-summary")
async def get_daily_summary():
    """Generate a daily summary."""
    try:
        summary = proactive_engine.generate_daily_summary()
        return {"summary": summary, "date": datetime.now().isoformat()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/suggestions")
async def get_suggestions():
    """Get proactive suggestions."""
    try:
        suggestions = proactive_engine.generate_proactive_suggestions()
        return {"suggestions": suggestions, "count": len(suggestions)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/patterns")
async def get_patterns():
    """Get detected patterns."""
    try:
        return proactive_engine.detect_patterns()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/check-in")
async def send_check_in():
    """Manually trigger a daily check-in."""
    try:
        return await scheduler.send_daily_check_in()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class LearnRequest(BaseModel):
    message: str


@router.post("/learn")
async def learn_from_message(request: LearnRequest):
    """Learn from a user message."""
    from app.services.feedback import FeedbackService

    try:
        feedback = FeedbackService()
        return await feedback.process_correction(request.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
