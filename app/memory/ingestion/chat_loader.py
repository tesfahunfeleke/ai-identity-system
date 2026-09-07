"""
Loader for JSON chat exports (ChatGPT, Claude, and a simple flat fallback).

Real-world chat exports vary a lot in shape:

  - ChatGPT's official export (conversations.json) uses a tree-structured
    "mapping" of message nodes with parent/child references, content buried
    in message.content.parts[], and Unix timestamps.
  - Claude's export uses a flatter structure: a list of conversations, each
    with a "chat_messages" list of {sender, text, created_at} objects.
  - Many third-party export tools (browser extensions, etc.) produce a
    simple flat {"messages": [{"role": ..., "content": ...}]} shape.

This loader detects which shape it's looking at and normalizes all of them
into the same RawChunk output -- one chunk per conversation, with the
back-and-forth flattened into a single readable transcript. Conversation-level
chunking (not per-message) is deliberate: a single exchange is usually the
right semantic unit for both fact extraction and later retrieval, and keeps
chunk count sane for long chat histories.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.memory.ingestion.base import BaseLoader
from app.memory.ingestion.models import RawChunk, SourceType


def _unix_to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value))
    except (TypeError, ValueError, OSError):
        return None


def _iso_to_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        # Claude exports typically use ISO 8601, e.g. "2026-03-14T10:00:00Z"
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class ChatExportLoader(BaseLoader):
    """Loads JSON chat exports from ChatGPT, Claude, or a simple flat format."""

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".json"

    def load(self, file_path: Path) -> list[RawChunk]:
        raw = json.loads(file_path.read_text(encoding="utf-8"))

        # Normalize to a list of "conversation" dicts regardless of top-level shape.
        conversations = raw if isinstance(raw, list) else [raw]

        chunks: list[RawChunk] = []
        for convo in conversations:
            if not isinstance(convo, dict):
                continue
            chunk = self._load_conversation(convo, file_path)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def _load_conversation(self, convo: dict, file_path: Path) -> RawChunk | None:
        """Detect the conversation's shape and dispatch to the right parser."""
        if "mapping" in convo:
            return self._parse_chatgpt_format(convo, file_path)
        if "chat_messages" in convo:
            return self._parse_claude_format(convo, file_path)
        if "messages" in convo:
            return self._parse_flat_format(convo, file_path)
        return None

    # --- ChatGPT official export format ---
    def _parse_chatgpt_format(self, convo: dict, file_path: Path) -> RawChunk | None:
        mapping: dict = convo.get("mapping", {})
        title = convo.get("title", "Untitled conversation")
        created = _unix_to_datetime(convo.get("create_time"))

        # Walk every node in the tree in the order ChatGPT stored them.
        # This is a simplification (true branch order would require following
        # parent/child links), but for fact-extraction purposes a
        # roughly-chronological flat transcript is sufficient.
        nodes = sorted(
            mapping.values(),
            key=lambda n: (n.get("message") or {}).get("create_time") or 0,
        )

        lines = []
        for node in nodes:
            message = node.get("message")
            if not message:
                continue
            role = (message.get("author") or {}).get("role", "unknown")
            content = message.get("content") or {}
            parts = content.get("parts") or []
            text = "\n".join(p for p in parts if isinstance(p, str)).strip()
            if not text:
                continue
            lines.append(f"{role.upper()}: {text}")

        if not lines:
            return None

        return RawChunk(
            text="\n\n".join(lines),
            source_type=SourceType.CHAT_EXPORT,
            source_file=str(file_path),
            original_date=created,
            extra={"title": title, "platform": "chatgpt"},
        )

    # --- Claude export format ---
    def _parse_claude_format(self, convo: dict, file_path: Path) -> RawChunk | None:
        title = convo.get("name") or convo.get("title") or "Untitled conversation"
        created = _iso_to_datetime(convo.get("created_at"))

        lines = []
        for msg in convo.get("chat_messages", []):
            sender = msg.get("sender", "unknown")
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"{sender.upper()}: {text}")

        if not lines:
            return None

        return RawChunk(
            text="\n\n".join(lines),
            source_type=SourceType.CHAT_EXPORT,
            source_file=str(file_path),
            original_date=created,
            extra={"title": title, "platform": "claude"},
        )

    # --- Simple flat fallback format: {"messages": [{"role", "content"}]} ---
    def _parse_flat_format(self, convo: dict, file_path: Path) -> RawChunk | None:
        title = convo.get("title", "Untitled conversation")
        created = _iso_to_datetime(convo.get("created_at")) or _unix_to_datetime(
            convo.get("create_time")
        )

        lines = []
        for msg in convo.get("messages", []):
            role = msg.get("role", "unknown")
            text = (msg.get("content") or "").strip()
            if not text:
                continue
            lines.append(f"{role.upper()}: {text}")

        if not lines:
            return None

        return RawChunk(
            text="\n\n".join(lines),
            source_type=SourceType.CHAT_EXPORT,
            source_file=str(file_path),
            original_date=created,
            extra={"title": title, "platform": "generic"},
        )
