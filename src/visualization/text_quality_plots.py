"""Text quality visualizations: WER/CER heatmaps, BLEU bar charts, LLM scatter."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

from .style import apply_style, save_fig, METHOD_COLORS, METHOD_LABELS, FIGURE_WIDTH_DOUBLE


FIXED_P = [0.01, 0.05, 0.10, 0.20, 0.30]


def plot_text_metrics_bar(
    rows: List[Dict[str, Any]],
    output_path: str = "figures/fig3_text_quality.pdf",
):
    apply_style()
    import pandas as pd
    df = pd.DataFrame(rows)

    p_sel = [p for p in FIXED_P if p in df["p"].values or any(abs(df["p"] - p) < 0.005)]
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_DOUBLE, 3.5))

    metrics = ["cer", "wer", "bleu"]
    titles = ["Character Error Rate", "Word Error Rate", "BLEU Score"]

    for ax, metric, title in zip(axes, metrics, titles):
        methods = [m for m in df["method"].unique() if "llm" not in m]
        x = np.arange(len(p_sel))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            vals = []
            for p in p_sel:
                sub = df[(df["method"] == method) & (abs(df["p"] - p) < 0.01)]
                vals.append(sub[metric].mean() if len(sub) else np.nan)
            ax.bar(
                x + i * width,
                vals,
                width,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, "gray"),
            )

        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"p={p}" for p in p_sel], rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1 if metric != "bleu" else 1.05)

    axes[0].legend(fontsize=7)
    fig.suptitle("Text Quality Metrics by Method and Channel Noise", fontsize=11)
    save_fig(fig, output_path)
    plt.close(fig)


def plot_llm_vs_ecc_scatter(
    ecc_rows: List[Dict[str, Any]],
    llm_rows: List[Dict[str, Any]],
    output_path: str = "figures/fig5_llm_vs_ecc.pdf",
):
    apply_style()
    import pandas as pd

    ecc_df = pd.DataFrame(ecc_rows)
    llm_df = pd.DataFrame(llm_rows)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    ecc_methods = [m for m in ecc_df["method"].unique() if m != "raw"]
    p_values = sorted(ecc_df["p"].unique())
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(p_values), max(p_values))

    for method in ecc_methods:
        for p in p_values:
            ecc_wer = ecc_df[(ecc_df["method"] == method) & (abs(ecc_df["p"] - p) < 0.005)]["wer"].mean()
            llm_wer = llm_df[abs(llm_df["p"] - p) < 0.005]["wer"].mean()
            if np.isnan(ecc_wer) or np.isnan(llm_wer):
                continue
            ax.scatter(
                ecc_wer, llm_wer,
                color=cmap(norm(p)),
                marker={"hamming": "s", "repetition_3": "^", "repetition_5": "v"}.get(method, "o"),
                s=60, alpha=0.8,
            )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ECC = LLM")
    ax.set_xlabel("ECC Word Error Rate")
    ax.set_ylabel("LLM Word Error Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="p value")
    ax.set_title("LLM vs ECC: Word Error Rate Comparison")

    save_fig(fig, output_path)
    plt.close(fig)
