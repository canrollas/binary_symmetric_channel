"""Reed-Solomon error correcting code over GF(2^8)."""

import numpy as np
from .base import ErrorCorrectingCode


class ReedSolomonCode(ErrorCorrectingCode):
    """
    RS(255, 255-nsym) code over GF(2^8).
    Corrects up to nsym//2 byte errors.
    Bit-level BSC corruption is grouped into bytes before RS decoding.
    """

    def __init__(self, nsym: int = 10):
        try:
            import reedsolo
        except ImportError:
            raise ImportError("Install reedsolo: pip install reedsolo")
        self.nsym = nsym
        self._codec = reedsolo.RSCodec(nsym)

    @property
    def name(self) -> str:
        return f"Reed-Solomon(nsym={self.nsym})"

    @property
    def rate(self) -> float:
        return (255 - self.nsym) / 255

    def encode_bits(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        # Pad bits to byte boundary
        remainder = len(bits) % 8
        if remainder:
            bits = np.pad(bits, (0, 8 - remainder))
        self._original_bit_len = len(bits)
        raw_bytes = np.packbits(bits).tobytes()
        encoded_bytes = bytes(self._codec.encode(raw_bytes))
        return np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8))

    def decode_bits(self, bits: np.ndarray) -> np.ndarray:
        import reedsolo
        bits = np.asarray(bits, dtype=np.uint8)
        remainder = len(bits) % 8
        if remainder:
            bits = np.pad(bits, (0, 8 - remainder))
        raw_bytes = np.packbits(bits).tobytes()
        try:
            decoded_bytes, _, _ = self._codec.decode(raw_bytes)
            decoded_bits = np.unpackbits(np.frombuffer(bytes(decoded_bytes), dtype=np.uint8))
        except reedsolo.ReedSolomonError:
            # Decoding failed — return as-is (byte-level)
            n_data = len(raw_bytes) - self.nsym
            if n_data > 0:
                decoded_bits = np.unpackbits(np.frombuffer(raw_bytes[:n_data], dtype=np.uint8))
            else:
                decoded_bits = np.zeros(getattr(self, "_original_bit_len", len(bits)), dtype=np.uint8)
        original_len = getattr(self, "_original_bit_len", len(decoded_bits))
        return decoded_bits[:original_len]
