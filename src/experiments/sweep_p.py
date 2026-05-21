"""Sweep p from 0.001 to 0.499 across all ECC methods."""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from .base_runner import ExperimentRunner, CORPUS, load_corpus
from ..channel.bsc import BSC
from ..ecc.hamming import HammingCode
from ..ecc.repetition import RepetitionCode
from ..ecc.reed_solomon import ReedSolomonCode
from ..hamming_bsc.utils import text_to_bits, bits_to_text
from ..metrics.bit_metrics import calculate_ber
from ..metrics.text_metrics import character_error_rate, word_error_rate, bleu_score
from ..metrics.shannon import shannon_capacity


def _process_raw(text: str, channel: BSC) -> Dict[str, float]:
    bits = text_to_bits(text)
    noisy, _ = channel.transmit(bits)
    recovered = bits_to_text(noisy)
    ber = calculate_ber(bits, noisy)
    return {
        "ber_pre": ber,
        "ber_post": ber,
        "cer": character_error_rate(text, recovered),
        "wer": word_error_rate(text, recovered),
        "bleu": bleu_score(text, recovered),
    }


def run_sweep(
    p_values: np.ndarray,
    n_trials: int = 10,
    seed_base: int = 42,
    output_dir: str = "results/sweep_p",
    corpus: Optional[List[str]] = None,
    use_rs: bool = True,
) -> List[Dict[str, Any]]:
    if corpus is None:
        corpus = load_corpus()

    codes = {
        "raw": None,
        "hamming": HammingCode,
        "repetition_3": lambda: RepetitionCode(3),
        "repetition_5": lambda: RepetitionCode(5),
    }
    if use_rs:
        codes["reed_solomon"] = lambda: ReedSolomonCode(10)

    all_rows: List[Dict[str, Any]] = []

    for p in tqdm(p_values, desc="p sweep"):
        for trial in range(n_trials):
            seed = seed_base + trial * 1000 + int(p * 1e6) % 1000
            channel = BSC(p, seed=seed)

            for method_name, code_factory in codes.items():
                metrics_list = []
                for text in corpus:
                    try:
                        if code_factory is None:
                            m = _process_raw(text, channel)
                        else:
                            code = code_factory()
                            m = code.process(text, channel)
                            m = {
                                "ber_pre": m["ber_pre"],
                                "ber_post": m["ber_post"],
                                "cer": m["cer"],
                                "wer": m["wer"],
                                "bleu": m["bleu"],
                            }
                        metrics_list.append(m)
                    except Exception:
                        pass

                if not metrics_list:
                    continue

                row = {
                    "p": float(p),
                    "method": method_name,
                    "trial": trial,
                    "capacity": float(shannon_capacity(p)),
                }
                for key in ["ber_pre", "ber_post", "cer", "wer", "bleu"]:
                    vals = [m[key] for m in metrics_list]
                    row[key] = float(np.mean(vals))
                all_rows.append(row)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/raw_results.json", "w") as f:
        json.dump(all_rows, f)
    print(f"Sweep done: {len(all_rows)} rows → {output_dir}/raw_results.json")
    return all_rows


class SweepPExperiment(ExperimentRunner):

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        p_values = np.linspace(
            cfg.get("p_start", 0.001),
            cfg.get("p_stop", 0.499),
            cfg.get("n_steps", 50),
        )
        rows = run_sweep(
            p_values=p_values,
            n_trials=cfg.get("n_trials", 10),
            seed_base=cfg.get("random_seed_base", 42),
            output_dir=cfg.get("output_dir", "results/sweep_p"),
            use_rs=cfg.get("use_rs", True),
        )
        return {"rows": rows, "n_rows": len(rows)}
