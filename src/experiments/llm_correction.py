"""LLM correction experiment across p values and prompt strategies."""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from .base_runner import ExperimentRunner, CORPUS, load_corpus
from ..channel.bsc import BSC
from ..hamming_bsc.utils import text_to_bits, bits_to_text
from ..llm.corrector import LLMCorrector
from ..metrics.text_metrics import character_error_rate, word_error_rate, bleu_score
from ..metrics.bit_metrics import calculate_ber
from ..metrics.shannon import shannon_capacity


def _row_key(strategy: str, p: float, trial: int) -> str:
    return f"{strategy}|{p:.6f}|{trial}"


def _load_existing(output_dir: str) -> Dict[str, Dict]:
    """Load already-completed rows from incremental file."""
    path = Path(output_dir) / "llm_results_incremental.jsonl"
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    key = _row_key(
                        row["method"].replace("llm_", ""),
                        row["p"],
                        row["trial"],
                    )
                    done[key] = row
                except Exception:
                    pass
    return done


def run_llm_experiment(
    p_values: List[float],
    prompt_strategies: List[str] = ("minimal", "contextual", "exemplar"),
    model: str = "anthropic/claude-sonnet-4-6",
    n_trials: int = 3,
    seed_base: int = 42,
    output_dir: str = "results/llm_correction",
    corpus: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if corpus is None:
        corpus = load_corpus(max_sentences=50)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    incremental_path = Path(output_dir) / "llm_results_incremental.jsonl"

    # Resume: load already-done rows
    done = _load_existing(output_dir)
    if done:
        print(f"  Resuming: {len(done)} rows already done, skipping.")

    all_rows: List[Dict[str, Any]] = list(done.values())

    for strategy in prompt_strategies:
        corrector = LLMCorrector(
            model=model,
            prompt_strategy=strategy,
            cache_path=f"{output_dir}/responses.jsonl",
        )

        for p in tqdm(p_values, desc=f"LLM [{strategy}]"):
            for trial in range(n_trials):
                key = _row_key(strategy, p, trial)
                if key in done:
                    continue  # already computed

                seed = seed_base + trial * 1000 + int(p * 1e6) % 1000
                channel = BSC(p, seed=seed)

                metrics_list = []
                for text in corpus:
                    bits = text_to_bits(text)
                    noisy, _ = channel.transmit(bits)
                    corrupted_text = bits_to_text(noisy)
                    recovered = corrector.correct(corrupted_text, p=p)

                    metrics_list.append({
                        "ber_pre": calculate_ber(bits, noisy),
                        "cer": character_error_rate(text, recovered),
                        "wer": word_error_rate(text, recovered),
                        "bleu": bleu_score(text, recovered),
                    })

                row = {
                    "p": float(p),
                    "method": f"llm_{strategy}",
                    "trial": trial,
                    "capacity": float(shannon_capacity(p)),
                }
                for k in ["ber_pre", "cer", "wer", "bleu"]:
                    row[k] = float(np.mean([m[k] for m in metrics_list]))

                all_rows.append(row)
                # Incremental save — crash-safe
                with open(incremental_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

        print(f"  Cost so far: ${corrector.total_cost_estimate():.4f}")

    # Final consolidated save
    with open(f"{output_dir}/llm_results.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Saved: {output_dir}/llm_results.json ({len(all_rows)} rows)")
    return all_rows


class LLMCorrectionExperiment(ExperimentRunner):

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        rows = run_llm_experiment(
            p_values=cfg.get("p_values", [0.01, 0.05, 0.10, 0.20, 0.30]),
            prompt_strategies=cfg.get("prompt_strategies", ["contextual"]),
            model=cfg.get("model", "anthropic/claude-sonnet-4-6"),
            n_trials=cfg.get("n_trials", 3),
            output_dir=cfg.get("output_dir", "results/llm_correction"),
        )
        return {"rows": rows}
