"""Abstract base for all experiment runners."""

import json
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List


def _load_corpus_file(path: str = "data/corpus.txt") -> List[str]:
    p = Path(path)
    if p.exists():
        lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            return lines
    return []


# Fallback seed corpus (used when data/corpus.txt has not been built yet)
CORPUS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Shannon proved that reliable communication is possible below channel capacity.",
    "Error correcting codes add redundancy to protect information against noise.",
    "In a binary symmetric channel, each bit is flipped with equal probability.",
    "The Hamming distance between two codewords determines the error correction capability.",
    "Reed-Solomon codes operate over Galois fields and correct burst errors efficiently.",
    "Large language models learn statistical patterns from vast amounts of text data.",
    "Information theory provides the mathematical foundation for modern communications.",
    "The entropy of a fair coin flip is exactly one bit of information.",
    "Turbo codes and LDPC codes approach the Shannon limit with practical complexity.",
    "Machine learning has transformed natural language processing in recent years.",
    "The capacity of a channel is the maximum rate at which information can be reliably transmitted.",
    "Majority voting is the simplest form of error correction through repetition.",
    "Claude can recover corrupted text by leveraging its training on natural language.",
    "The bit error rate measures the fraction of bits incorrectly received.",
    "Digital communications systems encode, transmit, and decode information continuously.",
    "Noise in communication channels is often modeled as a random process.",
    "Forward error correction eliminates the need for retransmission in one-way channels.",
    "The generator matrix of a linear code defines the mapping from message to codeword.",
    "Syndrome decoding uses parity check equations to identify error locations.",
    "Natural language has redundancy that allows humans to understand corrupted speech.",
    "The binary entropy function reaches its maximum of one bit at p equals one half.",
    "Wireless channels experience fading, multipath, and interference beyond simple BSC.",
    "Convolutional codes use sliding window encoding for continuous data streams.",
    "The probability of undetected error decreases exponentially with codeword length.",
    "Text compression exploits statistical redundancy to reduce bit representation.",
    "Hamming invented error correcting codes after experiencing frequent computer errors.",
    "The signal-to-noise ratio determines the practical channel capacity in analog systems.",
    "Modern deep learning decoders can learn code structure from examples automatically.",
    "Joint source-channel coding optimizes compression and error protection simultaneously.",
    "A repetition code of rate one third can correct any single error in a triple.",
    "The minimum distance of a code must exceed twice the number of correctable errors.",
    "Galois fields provide the algebraic structure needed for Reed-Solomon encoding.",
    "Channel polarization is the key insight behind polar codes achieving capacity.",
    "The parity check matrix is orthogonal to the generator matrix of any linear code.",
    "Burst errors that affect consecutive bits require different coding strategies.",
    "Interleaving spreads burst errors across multiple codewords for easier correction.",
    "The free distance of a convolutional code determines its error correcting power.",
    "Soft decision decoding uses probability information for better performance.",
    "Maximum likelihood decoding finds the codeword closest to the received sequence.",
]


class ExperimentRunner(ABC):

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else {}

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f)

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        ...

    def save_results(self, results: Dict[str, Any], output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out = Path(output_dir) / "summary.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out}")

    def get_corpus(self, corpus_path: str = "data/corpus.txt") -> List[str]:
        loaded = _load_corpus_file(corpus_path)
        return loaded if loaded else CORPUS


def load_corpus(path: str = "data/corpus.txt", max_sentences: int = 0) -> List[str]:
    """Public helper: load corpus file or fall back to seed sentences."""
    loaded = _load_corpus_file(path)
    corpus = loaded if loaded else CORPUS
    if max_sentences > 0:
        return corpus[:max_sentences]
    return corpus


from typing import Optional
