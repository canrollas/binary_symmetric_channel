"""ECC + LLM combined pipeline experiment."""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from .base_runner import ExperimentRunner, CORPUS
from ..channel.bsc import BSC
from ..ecc.hamming import HammingCode
from ..ecc.repetition import RepetitionCode
from ..hamming_bsc.utils import bits_to_text
from ..llm.corrector import LLMCorrector
from ..metrics.text_metrics import character_error_rate, word_error_rate, bleu_score
from ..metrics.shannon import shannon_capacity


def run_combined(
    p_values: List[float],
    model: str = "anthropic/claude-sonnet-4-6",
    n_trials: int = 3,
    seed_base: int = 42,
    output_dir: str = "results/combined",
    corpus: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if corpus is None:
        corpus = CORPUS[:20]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    corrector = LLMCorrector(model=model, cache_path=f"{output_dir}/responses.jsonl")

    ecc_options = {
        "hamming+llm": HammingCode,
        "repetition3+llm": lambda: RepetitionCode(3),
    }

    all_rows: List[Dict[str, Any]] = []

    for ecc_name, ecc_factory in ecc_options.items():
        for p in tqdm(p_values, desc=f"Combined [{ecc_name}]"):
            for trial in range(n_trials):
                seed = seed_base + trial * 1000 + int(p * 1e6) % 1000
                channel = BSC(p, seed=seed)

                metrics_list = []
                for text in corpus:
                    code = ecc_factory()
                    ecc_result = code.process(text, channel)
                    ecc_text = ecc_result["recovered"]
                    try:
                        final_text = corrector.correct(ecc_text, p=p)
                    except Exception:
                        final_text = ecc_text

                    metrics_list.append({
                        "cer_ecc": ecc_result["cer"],
                        "wer_ecc": ecc_result["wer"],
                        "bleu_ecc": ecc_result["bleu"],
                        "cer_combined": character_error_rate(text, final_text),
                        "wer_combined": word_error_rate(text, final_text),
                        "bleu_combined": bleu_score(text, final_text),
                    })

                row = {
                    "p": float(p),
                    "method": ecc_name,
                    "trial": trial,
                    "capacity": float(shannon_capacity(p)),
                }
                for key in ["cer_ecc", "wer_ecc", "bleu_ecc", "cer_combined", "wer_combined", "bleu_combined"]:
                    row[key] = float(np.mean([m[key] for m in metrics_list]))
                all_rows.append(row)

    with open(f"{output_dir}/combined_results.json", "w") as f:
        json.dump(all_rows, f)
    return all_rows


class CombinedExperiment(ExperimentRunner):

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        rows = run_combined(
            p_values=cfg.get("p_values", [0.01, 0.05, 0.10, 0.20]),
            model=cfg.get("model", "anthropic/claude-sonnet-4-6"),
            n_trials=cfg.get("n_trials", 3),
            output_dir=cfg.get("output_dir", "results/combined"),
        )
        return {"rows": rows}
