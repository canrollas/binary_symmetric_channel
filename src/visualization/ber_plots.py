"""BER vs p plots."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

from .style import apply_style, save_fig, METHOD_COLORS, METHOD_MARKERS, METHOD_LABELS, FIGURE_WIDTH_DOUBLE
from ..metrics.shannon import theoretical_hamming_ber, theoretical_repetition_ber


def plot_ber_vs_p(
    rows: List[Dict[str, Any]],
    output_path: str = "figures/fig1_ber_vs_p.pdf",
    log_scale: bool = True,
):
    apply_style()
    import pandas as pd
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_DOUBLE, 3.5))

    for ax, metric, title in zip(
        axes, ["ber_post", "cer"], ["Bit Error Rate (post-correction)", "Character Error Rate"]
    ):
        methods = df["method"].unique()
        for method in methods:
            sub = df[df["method"] == method].groupby("p")[metric].agg(["mean", "std"]).reset_index()
            color = METHOD_COLORS.get(method, "gray")
            marker = METHOD_MARKERS.get(method, "o")
            label = METHOD_LABELS.get(method, method)
            ax.plot(sub["p"], sub["mean"], color=color, marker=marker, label=label, markevery=5)
            ax.fill_between(
                sub["p"],
                sub["mean"] - sub["std"],
                sub["mean"] + sub["std"],
                alpha=0.15,
                color=color,
            )

        # Theoretical curves
        p_arr = np.linspace(0.001, 0.499, 200)
        ax.plot(p_arr, p_arr, "k--", lw=1, label="Theoretical raw", alpha=0.5)
        if metric == "ber_post":
            ax.plot(p_arr, [theoretical_hamming_ber(p) for p in p_arr], "b:", lw=1, label="Hamming theoretical")

        ax.set_xlabel("Bit-flip probability p")
        ax.set_ylabel(title)
        ax.set_xlim(0, 0.5)
        if log_scale and metric == "ber_post":
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1)
        else:
            ax.set_ylim(0, 1)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Error rates vs. channel noise", fontsize=12)
    save_fig(fig, output_path)
    plt.close(fig)
