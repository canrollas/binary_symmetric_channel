"""Run only LLM + Combined experiments and regenerate all figures."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from src.experiments.llm_correction import run_llm_experiment
from src.experiments.combined import run_combined
from src.visualization.ber_plots import plot_ber_vs_p
from src.visualization.capacity_plots import plot_shannon_bound
from src.visualization.text_quality_plots import plot_text_metrics_bar, plot_llm_vs_ecc_scatter
from src.visualization.summary_figure import plot_summary

# Load existing ECC sweep results
ecc_path = Path("results/sweep_p/raw_results.json")
if not ecc_path.exists():
    print("ERROR: results/sweep_p/raw_results.json bulunamadı. Önce ECC sweep çalıştırın.")
    sys.exit(1)

with open(ecc_path) as f:
    ecc_rows = json.load(f)
print(f"ECC sweep sonuçları yüklendi: {len(ecc_rows)} satır")

print("\n" + "="*60)
print("Step 2: LLM correction experiment")
print("="*60)
llm_rows = run_llm_experiment(
    p_values=[0.01, 0.05, 0.10, 0.20, 0.30],
    prompt_strategies=["minimal", "contextual", "exemplar"],
    n_trials=3,
    output_dir="results/llm_correction",
)

print("\n" + "="*60)
print("Step 3: Combined ECC + LLM experiment")
print("="*60)
combined_rows = run_combined(
    p_values=[0.01, 0.05, 0.10, 0.20],
    n_trials=3,
    output_dir="results/combined",
)

print("\nTüm figürler yeniden üretiliyor...")
plot_ber_vs_p(ecc_rows, "figures/fig1_ber_vs_p")
plot_shannon_bound(ecc_rows, "figures/fig2_shannon_bound")
plot_text_metrics_bar(ecc_rows, "figures/fig3_text_quality")
plot_llm_vs_ecc_scatter(ecc_rows, llm_rows, "figures/fig5_llm_vs_ecc")
plot_summary(ecc_rows, llm_rows, "figures/fig6_summary")

import shutil
Path("paper/figures").mkdir(exist_ok=True)
for f in Path("figures").glob("*.pdf"):
    shutil.copy(f, f"paper/figures/{f.name}")

print("\nBitti! Figürler figures/ ve paper/figures/ altında.")
