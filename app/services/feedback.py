import logging
import re
from typing import Any, Dict, Optional

from app.llm.client import llm_client
from app.memory.db import SessionLocal
from app.memory.facts import get_facts, write_fact

logger = logging.getLogger(__name__)


class FeedbackService:
    """Handles learning from corrections and feedback."""

    def __init__(self):
        self.correction_patterns = [
            r"actually[,]?\s+(.*?)(?:\.|$)",
            r"no,\s+(.*?)(?:\.|$)",
            r"that's?\s+(not|wrong|incorrect)\s+(.*?)(?:\.|$)",
            r"i\s+(?:really\s+)?(?:like|prefer|hate|dislike)\s+(.*?)(?:\.|$)",
        ]

    def extract_correction(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract correction information from a user message."""
        message_lower = message.lower()

        preference_match = re.search(
            r"(?:i\s+like|i\s+prefer|my\s+favorite)\s+(.*?)(?:\.|$)",
            message_lower,
        )
        if preference_match:
            return {
                "type": "preference",
                "original_text": message,
                "preference_text": preference_match.group(1),
                "confidence": 0.7,
            }

        for pattern in self.correction_patterns:
            match = re.search(pattern, message_lower)
            if match:
                correction_text = match.group(1)
                if correction_text:
                    return {
                        "type": "correction",
                        "original_text": message,
                        "correction_text": correction_text,
                        "confidence": 0.8,
                    }

        return None

    def learn_from_conversation(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """Learn from a conversation exchange."""
        result = {
            "learned": False,
            "facts_updated": [],
            "correction": None,
            "preference": None,
        }
        correction = self.extract_correction(user_message)
        if not correction:
            return result

        result["correction"] = correction
        result["learned"] = True
        if correction["type"] != "preference":
            return result

        try:
            with SessionLocal() as db:
                facts = get_facts(db)
                new_fact = {
                    "category": "preference",
                    "key": f"preference_{len(facts)}",
                    "value": correction["preference_text"],
                    "confidence": correction["confidence"],
                    "source_excerpt": user_message[:200],
                }
                fact = write_fact(db, **new_fact)
                result["facts_updated"].append({**new_fact, "id": fact.id})
                result["preference"] = new_fact
                logger.info("Learned new preference: %s", correction["preference_text"])
        except Exception as exc:
            logger.error("Error updating facts: %s", exc)
        return result

    async def process_correction(self, user_message: str) -> Dict[str, Any]:
        """Process a user correction and update memory."""
        result = {"processed": False, "message": "", "facts_updated": []}
        correction = self.extract_correction(user_message)
        if not correction:
            result["message"] = "No correction or preference detected."
            return result

        try:
            learned = self.learn_from_conversation(user_message, "")
            prompt = (
                f'The user said: "{user_message}"\n'
                "Extract the key fact as JSON with category, key, and value."
            )
            llm_client.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=150,
            )
            result["processed"] = True
            result["facts_updated"] = learned["facts_updated"]
            result["message"] = "✅ I've updated my memory with your correction! Thanks for helping me learn."
        except Exception as exc:
            logger.error("Error processing correction: %s", exc)
            result["message"] = "⚠️ I tried to learn from that, but had some trouble. Could you rephrase?"
        return result
