"""Orchestrate all experiments and generate figures."""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from src.experiments.sweep_p import run_sweep
from src.visualization.ber_plots import plot_ber_vs_p
from src.visualization.capacity_plots import plot_shannon_bound
from src.visualization.text_quality_plots import plot_text_metrics_bar
from src.visualization.summary_figure import plot_summary

import numpy as np


def main(skip_llm: bool = False, quick: bool = False):
    n_steps = 20 if quick else 60
    n_trials = 3 if quick else 10

    print("=" * 60)
    print("Step 1: ECC sweep across p values")
    print("=" * 60)
    p_values = np.linspace(0.001, 0.499, n_steps)
    rows = run_sweep(
        p_values=p_values,
        n_trials=n_trials,
        output_dir="results/sweep_p",
        use_rs=True,
    )

    print("\nGenerating figures...")
    plot_ber_vs_p(rows, "figures/fig1_ber_vs_p.pdf")
    plot_shannon_bound(rows, "figures/fig2_shannon_bound.pdf")
    plot_text_metrics_bar(rows, "figures/fig3_text_quality.pdf")

    llm_rows = []
    if not skip_llm:
        print("=" * 60)
        print("Step 2: LLM correction experiment")
        print("=" * 60)
        from src.experiments.llm_correction import run_llm_experiment
        llm_rows = run_llm_experiment(
            p_values=[0.01, 0.05, 0.10, 0.20, 0.30],
            prompt_strategies=["contextual"],
            n_trials=2 if quick else 3,
            output_dir="results/llm_correction",
        )

        print("=" * 60)
        print("Step 3: Combined ECC + LLM experiment")
        print("=" * 60)
        from src.experiments.combined import run_combined
        combined_rows = run_combined(
            p_values=[0.05, 0.10, 0.20],
            n_trials=2 if quick else 3,
            output_dir="results/combined",
        )

        from src.visualization.text_quality_plots import plot_llm_vs_ecc_scatter
        plot_llm_vs_ecc_scatter(rows, llm_rows, "figures/fig5_llm_vs_ecc.pdf")

    plot_summary(rows, llm_rows or None, "figures/fig6_summary.pdf")
    print("\nAll done. Figures in figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM experiments")
    parser.add_argument("--quick", action="store_true", help="Reduced trials for testing")
    args = parser.parse_args()
    main(skip_llm=args.skip_llm, quick=args.quick)
