"""Abstract base class for all Error Correcting Codes."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any
from ..channel.bsc import BSC


class ErrorCorrectingCode(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def rate(self) -> float: ...

    @abstractmethod
    def encode_bits(self, bits: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def decode_bits(self, bits: np.ndarray) -> np.ndarray: ...

    def encode_text(self, text: str) -> np.ndarray:
        from ..hamming_bsc.utils import text_to_bits
        return self.encode_bits(text_to_bits(text))

    def decode_text(self, bits: np.ndarray) -> str:
        from ..hamming_bsc.utils import bits_to_text
        return bits_to_text(self.decode_bits(bits))

    def process(self, text: str, channel: BSC) -> Dict[str, Any]:
        from ..hamming_bsc.utils import text_to_bits, bits_to_text
        from ..metrics.bit_metrics import calculate_ber
        from ..metrics.text_metrics import character_error_rate, word_error_rate, bleu_score

        original_bits = text_to_bits(text)
        encoded = self.encode_bits(original_bits)
        noisy, _ = channel.transmit(encoded)
        recovered_bits = self.decode_bits(noisy)
        recovered_text = bits_to_text(recovered_bits)

        # BER_pre: fraction of channel (encoded) bits flipped by the BSC
        ber_pre = calculate_ber(encoded, noisy)
        # BER_post: fraction of message bits wrong after decoding
        n = len(original_bits)
        ber_post = calculate_ber(original_bits, recovered_bits[:n] if len(recovered_bits) >= n else np.pad(recovered_bits, (0, n - len(recovered_bits))))

        return {
            "original": text,
            "recovered": recovered_text,
            "ber_pre": ber_pre,
            "ber_post": ber_post,
            "cer": character_error_rate(text, recovered_text),
            "wer": word_error_rate(text, recovered_text),
            "bleu": bleu_score(text, recovered_text),
            "overhead_ratio": len(encoded) / len(original_bits) if len(original_bits) > 0 else 1.0,
        }
