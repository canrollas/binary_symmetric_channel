"""LLM-based text correction via OpenRouter (OpenAI-compatible API)."""

import os
from typing import List, Dict, Any
from openai import OpenAI
from .cache import ResponseCache
from .prompt_builder import STRATEGIES, PromptStrategy


SYSTEM_PROMPT = (
    "You are a text restoration expert. Your ONLY job is to output the corrected text. "
    "DO NOT explain, DO NOT add markdown, DO NOT add quotes. "
    "Output exactly one line: the restored English sentence and nothing else."
)

DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _extract_corrected(raw) -> str:
    """Strip markdown/explanations — return the most sentence-like line."""
    import re
    if not raw:
        return ""
    raw = raw.strip()
    # Remove bold/italic markdown
    raw = re.sub(r'\*+([^*]+)\*+', r'\1', raw)
    # Split into lines, pick the longest that ends with punctuation
    lines = [l.strip().strip('"').strip("'") for l in raw.splitlines() if l.strip()]
    candidates = [l for l in lines if l and l[-1] in '.!?']
    if candidates:
        return max(candidates, key=len)
    # Fallback: longest non-empty line
    return max(lines, key=len) if lines else raw


def _make_client() -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


class LLMCorrector:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        prompt_strategy: str = "contextual",
        max_tokens: int = 512,
        cache_path: str = "results/llm_correction/responses.jsonl",
        use_cache: bool = True,
    ):
        self.client = _make_client()
        self.model = model
        self.strategy: PromptStrategy = STRATEGIES[prompt_strategy]
        self.max_tokens = max_tokens
        self.cache = ResponseCache(cache_path) if use_cache else None
        self.usage_log: List[Dict[str, Any]] = []

    def correct(self, corrupted_text: str, **kwargs) -> str:
        cache_key = ResponseCache.make_key(corrupted_text, self.model, self.strategy.name)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        user_messages = self.strategy.build(corrupted_text, **kwargs)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )

        result = _extract_corrected(response.choices[0].message.content)

        if response.usage:
            self.usage_log.append({
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            })

        if self.cache:
            self.cache.put(cache_key, result)

        return result

    def correct_batch(self, texts: List[str], **kwargs) -> List[str]:
        return [self.correct(t, **kwargs) for t in texts]

    def total_cost_estimate(self, input_mtok: float = 3.0, output_mtok: float = 15.0) -> float:
        total_in = sum(u["input_tokens"] for u in self.usage_log)
        total_out = sum(u["output_tokens"] for u in self.usage_log)
        return (total_in / 1e6) * input_mtok + (total_out / 1e6) * output_mtok
