from typing import Dict, Any, Optional
import os
from openai import AsyncOpenAI, OpenAI
from groq import Groq, AsyncGroq

from app.core.config import settings

class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.client = None
        self.async_client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = "gpt-4o-mini"
        elif self.provider == "groq":
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            self.async_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
            self.model = "openai/gpt-oss-120b"
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(self, messages: list, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return response.choices[0].message.content


llm_client = LLMClient()