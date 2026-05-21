"""
Build a research-grade corpus for BSC experiments.

Sources:
  1. Wikipedia API  — communication/information theory articles
  2. NLTK Brown corpus — general English (news, editorial, fiction, science)
  3. Hand-curated seed sentences (existing corpus)

Output: data/corpus.txt  — one sentence per line, UTF-8
        data/corpus_meta.json — source breakdown and statistics
"""

import re
import json
import random
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

# ── helpers ───────────────────────────────────────────────────────────────────

MIN_CHARS = 40
MAX_CHARS = 200
MIN_WORDS = 6
MAX_WORDS = 40


def is_clean(sentence: str) -> bool:
    s = sentence.strip()
    if not (MIN_CHARS <= len(s) <= MAX_CHARS):
        return False
    words = s.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False
    # Reject sentences that are mostly non-ASCII
    ascii_ratio = sum(1 for c in s if ord(c) < 128) / len(s)
    if ascii_ratio < 0.90:
        return False
    # Reject reference-heavy lines ([1], {{, ==)
    if re.search(r'\[\d+\]|\{\{|==|http|<|>', s):
        return False
    # Must end with sentence-ending punctuation
    if s[-1] not in '.!?':
        return False
    # Must have at least one lowercase letter (filters headings)
    if not any(c.islower() for c in s):
        return False
    return True


def split_into_sentences(text: str) -> List[str]:
    # Simple sentence splitter: split on '. ', '! ', '? '
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


# ── Source 1: Wikipedia ───────────────────────────────────────────────────────

WIKI_TOPICS = [
    "Binary_symmetric_channel",
    "Information_theory",
    "Shannon's_source_coding_theorem",
    "Hamming_code",
    "Reed–Solomon_error_correction",
    "Turbo_code",
    "Low-density_parity-check_code",
    "Polar_code_(coding_theory)",
    "Channel_capacity",
    "Entropy_(information_theory)",
    "Error_detection_and_correction",
    "Coding_theory",
    "Communication_channel",
    "Modulation",
    "Digital_signal_processing",
    "Natural_language_processing",
    "Large_language_model",
    "Transformer_(machine_learning_model)",
    "Machine_learning",
    "Deep_learning",
    "Computer_network",
    "Cryptography",
    "Data_compression",
    "Signal-to-noise_ratio",
    "Convolutional_code",
    "Viterbi_algorithm",
    "Forward_error_correction",
    "Cyclic_redundancy_check",
    "Parity_bit",
    "Galois_field",
    "Probability_theory",
    "Statistics",
    "Linear_algebra",
    "Digital_electronics",
    "Telecommunications",
]


def fetch_wikipedia(topic: str, max_sentences: int = 60, delay: float = 1.5) -> Tuple[List[str], str]:
    import time
    time.sleep(delay)
    try:
        import urllib.request
        import urllib.parse
        url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query",
                "titles": topic,
                "prop": "extracts",
                "explaintext": True,
                "exsectionformat": "plain",
                "format": "json",
                "redirects": 1,
            })
        )
        req = urllib.request.Request(url, headers={"User-Agent": "BSCResearch/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        sents = split_into_sentences(text)
        clean = [s for s in sents if is_clean(s)]
        return clean[:max_sentences], topic
    except Exception as e:
        print(f"  [WARN] Wikipedia {topic}: {e}")
        return [], topic


# ── Source 2: NLTK Brown Corpus ───────────────────────────────────────────────

BROWN_CATEGORIES = ["news", "editorial", "reviews", "science_fiction",
                    "learned", "government", "hobbies"]


def fetch_brown(max_sentences: int = 2000) -> List[str]:
    try:
        import nltk
        nltk.download("brown", quiet=True)
        nltk.download("punkt", quiet=True)
        from nltk.corpus import brown

        sentences = []
        for cat in BROWN_CATEGORIES:
            try:
                sents = brown.sents(categories=cat)
                for tokens in sents:
                    s = " ".join(tokens)
                    # Brown uses special markers — clean them
                    s = re.sub(r"\s([.,!?;:'])", r"\1", s)
                    s = re.sub(r'\s+', ' ', s).strip()
                    if is_clean(s):
                        sentences.append(s)
            except Exception:
                pass
        return sentences[:max_sentences]
    except ImportError:
        print("  [WARN] NLTK not installed — skipping Brown corpus")
        return []


# ── Source 3: Seed sentences (existing corpus) ────────────────────────────────

SEED_SENTENCES = [
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
    "The Viterbi algorithm finds the most likely path through a trellis diagram.",
]


# ── Main builder ──────────────────────────────────────────────────────────────

def build_corpus(
    output_path: str = "data/corpus.txt",
    meta_path: str = "data/corpus_meta.json",
    target_size: int = 5000,
    seed: int = 42,
    skip_wikipedia: bool = False,
    skip_brown: bool = False,
    append: bool = False,  # if True, merge with existing corpus.txt
):
    random.seed(seed)
    all_sentences: List[str] = []
    source_counts: dict = {}

    # 1. Seeds
    all_sentences.extend(SEED_SENTENCES)
    source_counts["seed"] = len(SEED_SENTENCES)
    print(f"[seed]      {len(SEED_SENTENCES)} sentences")

    # 2. Wikipedia
    wiki_sents = []
    if not skip_wikipedia:
        print(f"[wikipedia] Fetching {len(WIKI_TOPICS)} articles...")
        for topic in WIKI_TOPICS:
            sents, _ = fetch_wikipedia(topic, max_sentences=80)
            wiki_sents.extend(sents)
            print(f"  {topic}: {len(sents)} sentences")
    source_counts["wikipedia"] = len(wiki_sents)
    all_sentences.extend(wiki_sents)
    print(f"[wikipedia] {len(wiki_sents)} sentences total")

    # 3. Brown corpus
    if not skip_brown:
        print("[brown]     Fetching NLTK Brown corpus...")
        brown_sents = fetch_brown(max_sentences=3000)
        source_counts["brown"] = len(brown_sents)
        all_sentences.extend(brown_sents)
        print(f"[brown]     {len(brown_sents)} sentences")
    else:
        source_counts["brown"] = 0

    # Deduplicate
    seen = set()
    unique = []
    for s in all_sentences:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print(f"\nTotal unique sentences: {len(unique)}")

    # Shuffle and cap
    random.shuffle(unique)
    final = unique[:target_size]

    # Save corpus
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in final:
            f.write(s + "\n")

    # Save metadata
    lengths = [len(s) for s in final]
    word_counts = [len(s.split()) for s in final]
    meta = {
        "total_sentences": len(final),
        "sources": source_counts,
        "avg_chars": sum(lengths) / len(lengths),
        "avg_words": sum(word_counts) / len(word_counts),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "seed": seed,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCorpus saved: {output_path} ({len(final)} sentences)")
    print(f"Meta  saved: {meta_path}")
    print(f"Avg length: {meta['avg_chars']:.0f} chars, {meta['avg_words']:.1f} words")
    return final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--skip-wikipedia", action="store_true")
    parser.add_argument("--skip-brown", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_corpus(
        target_size=args.target,
        seed=args.seed,
        skip_wikipedia=args.skip_wikipedia,
        skip_brown=args.skip_brown,
    )
