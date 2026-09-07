import logging
import os
from typing import Any, Dict, List, Optional

from app.llm.client import llm_client
from app.memory.semantic_memory import SemanticMemoryService

logger = logging.getLogger(__name__)


class PersonalityEngine:
    """Handles personality-driven responses with emotional intelligence."""

    def __init__(self):
        self.memory_service = SemanticMemoryService()
        self.personality_profile = self._load_personality_profile()
        self.emotional_state = self._default_emotional_state()
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10

    def _load_personality_profile(self) -> str:
        prompt_file = os.getenv("PERSONALITY_SYSTEM_PROMPT_FILE", "app/llm/prompts/base.txt")
        try:
            with open(prompt_file, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            logger.warning("Personality file not found: %s", prompt_file)
            return "You are a helpful AI assistant."

    def _default_emotional_state(self) -> Dict[str, Any]:
        return {
            "mood": "neutral",
            "energy": 0.5,
            "sentiment": 0.0,
            "empathy": 0.5,
            "tone": "warm",
        }

    def detect_emotion(self, text: str) -> Dict[str, Any]:
        emotions = {
            "happy": ["happy", "great", "good", "wonderful", "excited", "amazing", "love"],
            "sad": ["sad", "unhappy", "depressed", "lonely", "crying", "miss", "disappointed"],
            "angry": ["angry", "mad", "frustrated", "annoyed", "upset", "irritated"],
            "anxious": ["anxious", "worried", "nervous", "stress", "overwhelmed"],
            "excited": ["excited", "thrilled", "pumped", "enthusiastic"],
            "tired": ["tired", "exhausted", "sleepy", "drained"],
            "grateful": ["grateful", "thankful", "appreciate", "blessed"],
            "curious": ["curious", "wonder", "interested", "learning"],
        }
        detected_emotions = [
            emotion for emotion, keywords in emotions.items()
            if any(keyword in text.lower() for keyword in keywords)
        ]
        primary = detected_emotions[0] if detected_emotions else "neutral"
        self.emotional_state["mood"] = primary
        self.emotional_state["sentiment"] = self._calculate_sentiment(text)
        tone_map = {
            "happy": "warm and cheerful", "sad": "gentle and comforting",
            "angry": "calm and understanding", "anxious": "reassuring and calm",
            "excited": "energetic and supportive", "tired": "gentle and restful",
            "grateful": "warm and appreciative", "curious": "engaged and thoughtful",
            "neutral": "warm and approachable",
        }
        self.emotional_state["tone"] = tone_map.get(primary, "warm and approachable")
        return self.emotional_state

    def _calculate_sentiment(self, text: str) -> float:
        positive_words = ["good", "great", "awesome", "love", "happy", "wonderful", "excellent", "nice", "best", "amazing"]
        negative_words = ["bad", "terrible", "awful", "hate", "sad", "worst", "poor", "disappointed", "frustrated", "stress"]
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        total = positive_count + negative_count
        return 0.0 if total == 0 else (positive_count - negative_count) / total

    def build_prompt(self, query: str, context: Optional[str] = None) -> str:
        emotional_state = self.detect_emotion(query)
        try:
            memories = self.memory_service.retrieve(query=query, k=3)
            memory_text = "\n".join(f"- {memory['text']}" for memory in memories) if memories else "No specific memories found."
        except Exception as exc:
            logger.warning("Error retrieving memories: %s", exc)
            memory_text = "No specific memories found."
        optional_context = f"Context: {context}" if context else ""
        return f"""{self.personality_profile}

Current emotional state of the user:
- Mood: {emotional_state['mood']}
- Sentiment: {emotional_state['sentiment']:.2f}
- Tone: {emotional_state['tone']}

Relevant memories about the user:
{memory_text}

User's question: {query}

{optional_context}

Your response should reflect your personality as the user's digital twin, show empathy, reference relevant memories when appropriate, use a {emotional_state['tone']} tone, and be natural and conversational.
"""

    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        prompt = self.build_prompt(query, context)
        messages = [{"role": "system", "content": prompt}]
        messages.extend(
            {"role": "user", "content": item["user"]} for item in self.conversation_history
        )
        messages.append({"role": "user", "content": query})
        try:
            response = llm_client.generate(messages, temperature=0.75)
            self.conversation_history.append({"user": query, "assistant": response})
            self.conversation_history = self.conversation_history[-self.max_history:]
            return response
        except Exception as exc:
            logger.error("Error generating response: %s", exc)
            return f"I'm having trouble generating a response right now. Error: {exc}"

    def get_emotional_state(self) -> Dict[str, Any]:
        return self.emotional_state

    def reset_conversation(self):
        self.conversation_history = []
        self.emotional_state = self._default_emotional_state()


personality_engine = PersonalityEngine()