"""Tests for LLM corrector (mocked — no real API calls)."""

import sys
import pytest
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock
from src.llm.prompt_builder import MinimalPrompt, ContextualPrompt, ExemplarPrompt
from src.llm.cache import ResponseCache


def test_minimal_prompt_returns_messages():
    p = MinimalPrompt()
    msgs = p.build("Hello wrold")
    assert isinstance(msgs, list)
    assert msgs[0]["role"] == "user"
    assert "Hello wrold" in msgs[0]["content"]


def test_contextual_prompt_includes_p():
    p = ContextualPrompt()
    msgs = p.build("Hello wrold", p=0.05)
    assert "0.0500" in msgs[0]["content"]


def test_exemplar_prompt_has_examples():
    p = ExemplarPrompt()
    msgs = p.build("Hello wrold")
    assert "Corrupted:" in msgs[0]["content"]


def test_cache_stores_and_retrieves(tmp_path):
    cache_file = str(tmp_path / "cache.jsonl")
    cache = ResponseCache(cache_file)
    key = ResponseCache.make_key("hello", "model-x", "minimal")
    assert cache.get(key) is None
    cache.put(key, "corrected")
    assert cache.get(key) == "corrected"


def test_cache_persists(tmp_path):
    cache_file = str(tmp_path / "cache.jsonl")
    key = ResponseCache.make_key("hello", "model-x", "minimal")
    c1 = ResponseCache(cache_file)
    c1.put(key, "corrected")
    c2 = ResponseCache(cache_file)
    assert c2.get(key) == "corrected"


def test_corrector_uses_cache(tmp_path, monkeypatch):
    """Corrector should not call API when result is already cached."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    cache_file = str(tmp_path / "responses.jsonl")
    cache = ResponseCache(cache_file)
    strategy = "contextual"
    model = "anthropic/claude-sonnet-4.6"
    key = ResponseCache.make_key("corrupted text", model, strategy)
    cache.put(key, "corrected text")

    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        from src.llm.corrector import LLMCorrector
        corrector = LLMCorrector(
            model=model,
            prompt_strategy=strategy,
            cache_path=cache_file,
            use_cache=True,
        )
        result = corrector.correct("corrupted text")

    mock_client.chat.completions.create.assert_not_called()
    assert result == "corrected text"
