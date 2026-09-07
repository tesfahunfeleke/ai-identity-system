import logging
from datetime import datetime
from typing import Any, Dict, List

from app.llm.client import llm_client
from app.memory.semantic_memory import SemanticMemoryService

logger = logging.getLogger(__name__)


class ProactiveEngine:
    """Handles proactive intelligence, habit tracking, and pattern detection."""

    def __init__(self):
        self.memory_service = SemanticMemoryService()
        self.conversation_history: List[Dict[str, str]] = []
        self.last_summary_date = None

    def detect_habits(self, text: str) -> List[Dict[str, Any]]:
        """Detect habits from text input."""
        habits = []
        habit_patterns = [
            ("running", "exercise"), ("run", "exercise"), ("gym", "exercise"), ("walk", "exercise"),
            ("meditate", "wellness"), ("journal", "writing"), ("read", "learning"),
            ("study", "learning"), ("practice", "skill"), ("write", "writing"),
            ("code", "programming"), ("draw", "creative"), ("paint", "creative"),
            ("cook", "cooking"), ("bake", "cooking"), ("plan", "planning"),
            ("review", "reflection"),
        ]
        text_lower = text.lower()
        for keyword, category in habit_patterns:
            if keyword in text_lower:
                habits.append({
                    "habit": keyword,
                    "category": category,
                    "detected_at": datetime.now().isoformat(),
                })
        return habits

    def track_habit(self, habit: str, category: str) -> Dict[str, Any]:
        """Track a habit occurrence."""
        return {
            "habit": habit,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "status": "tracked",
        }

    def detect_patterns(self) -> Dict[str, Any]:
        """Detect patterns from the user's data."""
        patterns = {
            "weekly_habits": {},
            "daily_routines": {},
            "preferences": {},
            "goals": [],
        }
        try:
            results = self.memory_service.retrieve(query="habits routines daily activities", k=5)
            for result in results:
                text = result.get("text", "").lower()
                if "morning" in text:
                    patterns["daily_routines"]["morning"] = "found"
                if "evening" in text or "night" in text:
                    patterns["daily_routines"]["evening"] = "found"
                if "week" in text or "weekend" in text:
                    patterns["weekly_habits"]["weekly"] = "found"
        except Exception as exc:
            logger.warning("Error detecting patterns: %s", exc)
        return patterns

    def detect_goals(self, text: str) -> List[Dict[str, Any]]:
        """Detect goals from text input."""
        goals = []
        goal_keywords = ["goal", "aim", "target", "want to", "plan to", "hope to", "working on"]
        text_lower = text.lower()
        for keyword in goal_keywords:
            if keyword in text_lower:
                for sentence in text.split("."):
                    if keyword in sentence.lower():
                        goals.append({
                            "goal": sentence.strip(),
                            "detected_at": datetime.now().isoformat(),
                        })
        return goals

    def generate_daily_summary(self) -> str:
        """Generate a daily summary of the user's activities."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            results = self.memory_service.retrieve(query=f"today {today} daily activities", k=5)
            if not results:
                return "No activities recorded for today yet."
            activity_text = "\n".join(f"- {result['text']}" for result in results)
            summary_prompt = f"""Based on the following activities, generate a brief daily summary:

{activity_text}

Include key activities, habits tracked, goals mentioned, and notable insights. Keep it concise and positive."""
            summary = llm_client.generate(
                [{"role": "user", "content": summary_prompt}],
                temperature=0.5,
                max_tokens=300,
            )
            self.last_summary_date = datetime.now()
            return summary
        except Exception as exc:
            logger.error("Error generating daily summary: %s", exc)
            return "Could not generate daily summary."

    def generate_proactive_suggestions(self) -> List[str]:
        """Generate proactive suggestions based on patterns."""
        suggestions = []
        try:
            patterns = self.detect_patterns()
            if patterns.get("daily_routines", {}).get("morning"):
                suggestions.append("Your morning routine seems consistent. Keep it up!")
            if patterns.get("preferences", {}).get("coffee"):
                suggestions.append("I noticed you enjoy coffee - have you tried any new blends lately?")
            results = self.memory_service.retrieve(query="goal plan want to", k=3)
            for result in results:
                text = result.get("text", "")
                if any(keyword in text.lower() for keyword in ["goal", "want to", "plan to"]):
                    suggestions.append(f"You mentioned working on: {text[:100]}... Would you like me to help track progress?")
        except Exception as exc:
            logger.warning("Error generating suggestions: %s", exc)
        return suggestions

    def get_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about the user."""
        return {
            "patterns": self.detect_patterns(),
            "suggestions": self.generate_proactive_suggestions(),
            "habits": [],
            "goals": [],
            "last_summary": self.last_summary_date,
        }
