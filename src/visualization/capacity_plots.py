"""Shannon capacity overlay plots."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

from .style import apply_style, save_fig, METHOD_COLORS, METHOD_LABELS, FIGURE_WIDTH_DOUBLE
from ..metrics.shannon import capacity_curve, shannon_capacity, binary_entropy


CODE_RATES = {
    "Hamming(7,4)": 4 / 7,
    "Repetition-3": 1 / 3,
    "Repetition-5": 1 / 5,
    "Reed-Solomon": (255 - 10) / 255,
}


def plot_shannon_bound(
    rows: List[Dict[str, Any]],
    output_path: str = "figures/fig2_shannon_bound.pdf",
):
    apply_style()
    import pandas as pd
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE * 0.65, 3.8))

    p_arr = np.linspace(0.001, 0.499, 300)
    cap = capacity_curve(p_arr)
    ax.fill_between(p_arr, 0, cap, alpha=0.07, color="black", label="Achievable region")
    ax.plot(p_arr, cap, "k-", lw=2.5, label="Shannon capacity C(p)")

    # Code rate horizontal lines
    colors = ["#0077BB", "#33BBEE", "#009988", "#EE7733"]
    for (name, rate), color in zip(CODE_RATES.items(), colors):
        ax.axhline(rate, color=color, ls="--", lw=1.2, label=f"{name} (rate={rate:.3f})")

    # Empirical achieved rates: 1 - H(ber_post)
    for method in df["method"].unique():
        if "llm" in method:
            continue
        sub = df[df["method"] == method].groupby("p")["ber_post"].mean().reset_index()
        color = METHOD_COLORS.get(method, "gray")
        label = METHOD_LABELS.get(method, method)
        achieved = sub["ber_post"].apply(lambda ber: 1 - binary_entropy(min(ber, 0.4999)))
        ax.plot(sub["p"], achieved, color=color, lw=1.5, alpha=0.8, marker=".", markersize=3)

    ax.set_xlabel("Bit-flip probability p")
    ax.set_ylabel("Information rate (bits/channel use)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Shannon Bound vs. Code Rates")

    save_fig(fig, output_path)
    plt.close(fig)
