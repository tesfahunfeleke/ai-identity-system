"""Tests for ChatExportLoader, covering ChatGPT, Claude, and generic flat formats."""

from pathlib import Path

from app.memory.ingestion.chat_loader import ChatExportLoader
from app.memory.ingestion.models import SourceType

FIXTURES = Path(__file__).parent / "fixtures" / "sample_journals"


def test_can_handle_json_only():
    loader = ChatExportLoader()
    assert loader.can_handle(Path("export.json"))
    assert not loader.can_handle(Path("export.txt"))


def test_loads_chatgpt_format():
    loader = ChatExportLoader()
    chunks = loader.load(FIXTURES / "chatgpt_export.json")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_type == SourceType.CHAT_EXPORT
    assert chunk.extra["platform"] == "chatgpt"
    assert chunk.extra["title"] == "Career thoughts"
    assert "USER:" in chunk.text and "ASSISTANT:" in chunk.text
    assert "creative freedom" in chunk.text
    assert chunk.original_date is not None


def test_loads_claude_format():
    loader = ChatExportLoader()
    chunks = loader.load(FIXTURES / "claude_export.json")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.extra["platform"] == "claude"
    assert "HUMAN:" in chunk.text and "ASSISTANT:" in chunk.text
    assert "favorite season is autumn" in chunk.text


def test_loads_generic_flat_format():
    loader = ChatExportLoader()
    chunks = loader.load(FIXTURES / "generic_chat.json")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.extra["platform"] == "generic"
    assert "USER:" in chunk.text
    assert "oat milk flat white" in chunk.text


def test_conversation_with_no_text_produces_no_chunk(tmp_path):
    empty_export = tmp_path / "empty.json"
    empty_export.write_text('[{"title": "Empty", "messages": []}]')
    loader = ChatExportLoader()
    assert loader.load(empty_export) == []
