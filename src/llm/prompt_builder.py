"""Prompt construction strategies for LLM text correction."""

from abc import ABC, abstractmethod
from typing import List, Dict


class PromptStrategy(ABC):
    name: str

    @abstractmethod
    def build(self, corrupted_text: str, **kwargs) -> List[Dict]:
        ...


class MinimalPrompt(PromptStrategy):
    name = "minimal"

    def build(self, corrupted_text: str, **kwargs) -> List[Dict]:
        return [
            {
                "role": "user",
                "content": (
                    "The following text was corrupted by a noisy binary channel. "
                    "Restore it to the most likely original English text. "
                    "Return ONLY the corrected text, nothing else.\n\n"
                    f"Corrupted: {corrupted_text}"
                ),
            }
        ]


class ContextualPrompt(PromptStrategy):
    name = "contextual"

    def build(self, corrupted_text: str, p: float = 0.05, **kwargs) -> List[Dict]:
        return [
            {
                "role": "user",
                "content": (
                    f"Text was transmitted through a Binary Symmetric Channel with bit-flip probability p={p:.4f}. "
                    f"Each bit was independently flipped with this probability, corrupting the original ASCII text. "
                    "Restore the most likely original English text. "
                    "Return ONLY the corrected text, nothing else.\n\n"
                    f"Corrupted: {corrupted_text}"
                ),
            }
        ]


class ExemplarPrompt(PromptStrategy):
    name = "exemplar"

    EXAMPLES = [
        ("Thd quiok brpwn fox", "The quick brown fox"),
        ("Hellp wprld!", "Hello world!"),
    ]

    def build(self, corrupted_text: str, **kwargs) -> List[Dict]:
        examples = "\n".join(
            f"Corrupted: {c}\nOriginal: {o}" for c, o in self.EXAMPLES
        )
        return [
            {
                "role": "user",
                "content": (
                    "Correct text corrupted by a noisy binary channel. Examples:\n\n"
                    f"{examples}\n\n"
                    "Now correct the following. Return ONLY the corrected text.\n\n"
                    f"Corrupted: {corrupted_text}"
                ),
            }
        ]


STRATEGIES = {
    "minimal": MinimalPrompt(),
    "contextual": ContextualPrompt(),
    "exemplar": ExemplarPrompt(),
}
