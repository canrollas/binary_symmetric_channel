"""Six-panel summary figure for the paper."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

from .style import apply_style, save_fig, METHOD_COLORS, METHOD_MARKERS, METHOD_LABELS, FIGURE_WIDTH_DOUBLE
from ..metrics.shannon import capacity_curve, binary_entropy


def plot_summary(
    rows: List[Dict[str, Any]],
    llm_rows: List[Dict[str, Any]] = None,
    output_path: str = "figures/fig6_summary.pdf",
):
    apply_style()
    import pandas as pd

    df = pd.DataFrame(rows)
    if llm_rows:
        llm_df = pd.DataFrame(llm_rows)
        combined = pd.concat([df, llm_df], ignore_index=True)
    else:
        combined = df

    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_DOUBLE, 6.5))
    axes = axes.flatten()

    p_arr = np.linspace(0.001, 0.499, 200)
    cap = capacity_curve(p_arr)

    # Panel A: BER vs p
    ax = axes[0]
    for method in combined["method"].unique():
        sub = combined[combined["method"] == method].groupby("p")["ber_post"].mean().reset_index()
        ax.plot(sub["p"], sub["ber_post"],
                color=METHOD_COLORS.get(method, "gray"),
                marker=METHOD_MARKERS.get(method, "o"),
                label=METHOD_LABELS.get(method, method),
                markevery=5)
    ax.set_yscale("log")
    ax.set_xlabel("p"); ax.set_ylabel("BER"); ax.set_title("(A) BER vs p")

    # Panel B: Shannon capacity
    ax = axes[1]
    ax.fill_between(p_arr, 0, cap, alpha=0.08, color="black")
    ax.plot(p_arr, cap, "k-", lw=2, label="Shannon C(p)")
    for method in df["method"].unique():
        if method == "raw":
            continue
        sub = df[df["method"] == method].groupby("p")["ber_post"].mean().reset_index()
        achieved = sub["ber_post"].apply(lambda b: 1 - binary_entropy(min(b, 0.4999)))
        ax.plot(sub["p"], achieved,
                color=METHOD_COLORS.get(method, "gray"),
                label=METHOD_LABELS.get(method, method))
    ax.set_xlabel("p"); ax.set_ylabel("Rate (bits/use)"); ax.set_title("(B) Shannon Bound")

    # Panels C, D, E: bar charts at fixed p
    fixed_p = [0.05, 0.10, 0.20]
    metrics = ["cer", "wer", "bleu"]
    panel_titles = ["(C) CER at fixed p", "(D) WER at fixed p", "(E) BLEU at fixed p"]

    for ax, metric, title in zip(axes[2:5], metrics, panel_titles):
        methods = list(combined["method"].unique())
        x = np.arange(len(fixed_p))
        w = 0.8 / len(methods)
        for i, method in enumerate(methods):
            vals = [combined[(combined["method"] == method) & (abs(combined["p"] - p) < 0.015)][metric].mean()
                    for p in fixed_p]
            ax.bar(x + i * w, vals, w, color=METHOD_COLORS.get(method, "gray"), label=METHOD_LABELS.get(method, method))
        ax.set_xticks(x + w * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"p={p}" for p in fixed_p])
        ax.set_title(title)
        ax.set_ylim(0, 1)

    # Panel F: LLM vs ECC scatter
    ax = axes[5]
    if llm_rows:
        for p in fixed_p:
            for method in ["hamming", "repetition_3"]:
                ecc_wer = df[(df["method"] == method) & (abs(df["p"] - p) < 0.015)]["wer"].mean()
                llm_wer = llm_df[abs(llm_df["p"] - p) < 0.015]["wer"].mean()
                if not (np.isnan(ecc_wer) or np.isnan(llm_wer)):
                    ax.scatter(ecc_wer, llm_wer, s=60,
                               color=METHOD_COLORS.get(method, "gray"),
                               marker=METHOD_MARKERS.get(method, "o"))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("ECC WER"); ax.set_ylabel("LLM WER")
    ax.set_title("(F) LLM vs ECC (WER)")

    axes[0].legend(fontsize=6, loc="upper left")
    axes[1].legend(fontsize=6, loc="upper right")

    fig.suptitle("BSC Correction: ECC vs LLM — Summary", fontsize=12, y=1.01)
    save_fig(fig, output_path)
    plt.close(fig)
