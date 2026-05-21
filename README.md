# Error Correction vs. Language Models on the Binary Symmetric Channel

**A Systematic Empirical Comparison**

> Baris Atakan (Prof. Dr.) · Burcu Baris (PhD Cand.) · Can Rollas (MSc Cand.) · Arda Tanil Kersu (MSc Cand.)  
> Izmir Institute of Technology

---

## Overview

This repository contains the full research codebase for our paper studying how classical **Error Correcting Codes (ECC)** and **Large Language Models (LLM)** compare when recovering text transmitted over a **Binary Symmetric Channel (BSC)**.

A BSC flips each transmitted bit independently with probability *p*. Shannon's channel coding theorem establishes the capacity:

```
C(p) = 1 - H(p) = 1 + p·log₂(p) + (1-p)·log₂(1-p)
```

We ask: **how much corrupted text can be recovered, and can LLMs exceed the Shannon bound?**

---

## Methods Compared

| Method | Code Rate | Corrects |
|---|---|---|
| Raw BSC (baseline) | 1 | — |
| Hamming(7,4) | 4/7 ≈ 0.571 | 1-bit errors per block |
| Repetition-3 | 1/3 | Majority vote (triples) |
| Repetition-5 | 1/5 | Majority vote (quintuples) |
| Reed-Solomon (nsym=10) | 245/255 ≈ 0.96 | Up to 5 byte errors per block |
| LLM (Claude Sonnet) | — | Semantic recovery via language prior |
| Hamming + LLM | — | ECC then LLM post-correction |

---

## Repository Structure

```
binary_symmetric_channel/
├── src/
│   ├── channel/          # BSC implementation
│   ├── ecc/              # Hamming, Repetition, Reed-Solomon
│   ├── llm/              # LLM corrector (OpenRouter), prompt strategies, cache
│   ├── metrics/          # BER, CER, WER, BLEU, Shannon capacity
│   ├── experiments/      # Experiment runners (sweep, LLM, combined)
│   └── visualization/    # Publication-quality figure generation
├── experiments/
│   ├── run_all.py        # Full pipeline (ECC + LLM + figures)
│   ├── run_llm.py        # LLM-only run (resumes from checkpoint)
│   └── configs/          # YAML configs per experiment
├── results/              # JSON outputs from experiments
├── figures/              # Generated figures (PDF + PNG)
├── data/
│   ├── corpus.txt        # 3040 sentences (Brown corpus + seeds)
│   └── build_corpus.py   # Corpus builder script
├── paper/                # LaTeX paper (IEEEtran format)
│   ├── main.tex
│   ├── sections/         # Introduction through Conclusion
│   └── bibliography.bib
└── tests/                # 29 unit tests
```

---

## Quickstart

### Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your OpenRouter API key:

```bash
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

### Run ECC Experiments (no API key needed)

```bash
.venv/bin/python experiments/run_all.py --skip-llm
```

Sweeps *p* from 0.001 to 0.499 across all ECC methods, generates figures.

### Run Full Pipeline (ECC + LLM)

```bash
.venv/bin/python experiments/run_all.py
```

### Resume LLM Experiments After Interruption

```bash
.venv/bin/python experiments/run_llm.py
```

LLM responses are cached in `results/llm_correction/responses.jsonl` — re-runs cost nothing for already-computed results. Intermediate rows are saved after each (strategy, p, trial) combination so crashes don't lose progress.

### Build the Paper

```bash
cd paper && make
```

Requires a LaTeX installation (`brew install --cask mactex` on macOS).

---

## Metrics

| Metric | Level | Description |
|---|---|---|
| BER | Bit | Fraction of bits wrong after decoding |
| CER | Character | Edit distance / original length |
| WER | Word | Word-level edit distance / word count |
| BLEU | Semantic | N-gram overlap with brevity penalty |

All results are benchmarked against the **Shannon capacity curve** *C(p)*.

---

## Key Findings (Preliminary)

- ECC methods respect the Shannon bound strictly — performance degrades exactly at the theoretical threshold
- LLM correction exploits English language redundancy (~1 bit/char vs. 8 bits/char raw), enabling semantic recovery well above the ECC failure point
- This does **not** violate Shannon's theorem — the LLM acts as a joint source-channel decoder using the source distribution as prior information
- The combined **ECC + LLM pipeline** outperforms either approach alone across all tested *p* values

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

29 tests covering channel simulation, all ECC methods, LLM caching, and metrics.

---

## Citation

```bibtex
@article{atakan2026bsc,
  title   = {Error Correction vs. Language Models on the Binary Symmetric Channel:
             A Systematic Empirical Comparison},
  author  = {Atakan, Baris and Baris, Burcu and Rollas, Can and Kersu, Arda Tanil},
  journal = {arXiv preprint},
  year    = {2026}
}
```
