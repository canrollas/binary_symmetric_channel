"""Centralized matplotlib style for publication figures."""

import matplotlib.pyplot as plt
import matplotlib as mpl

FIGURE_WIDTH_SINGLE = 3.5   # inches, one-column
FIGURE_WIDTH_DOUBLE = 7.2   # inches, two-column

# Okabe-Ito colorblind-safe palette
METHOD_COLORS = {
    "raw":            "#999999",
    "hamming":        "#0077BB",
    "repetition_3":   "#33BBEE",
    "repetition_5":   "#009988",
    "reed_solomon":   "#EE7733",
    "llm_minimal":    "#CC3311",
    "llm_contextual": "#EE3377",
    "llm_exemplar":   "#AA3377",
    "llm":            "#CC3311",
    "hamming+llm":    "#882255",
    "repetition3+llm":"#44AA99",
}

METHOD_MARKERS = {
    "raw":            "o",
    "hamming":        "s",
    "repetition_3":   "^",
    "repetition_5":   "v",
    "reed_solomon":   "D",
    "llm_minimal":    "x",
    "llm_contextual": "P",
    "llm_exemplar":   "*",
    "llm":            "P",
    "hamming+llm":    "H",
    "repetition3+llm":"X",
}

METHOD_LABELS = {
    "raw":            "No ECC (raw BSC)",
    "hamming":        "Hamming(7,4)",
    "repetition_3":   "Repetition-3",
    "repetition_5":   "Repetition-5",
    "reed_solomon":   "Reed-Solomon",
    "llm_minimal":    "LLM (minimal)",
    "llm_contextual": "LLM (contextual)",
    "llm_exemplar":   "LLM (exemplar)",
    "llm":            "LLM",
    "hamming+llm":    "Hamming + LLM",
    "repetition3+llm":"Repetition-3 + LLM",
}


def save_fig(fig, output_path: str, dpi: int = 150):
    """Save figure as both PDF and PNG."""
    from pathlib import Path
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = p.with_suffix(".pdf")
    png_path = p.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved: {pdf_path} + {png_path}")


def apply_style(use_latex: bool = False):
    mpl.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "figure.constrained_layout.use": True,
    })
    if use_latex:
        try:
            mpl.rcParams["text.usetex"] = True
            mpl.rcParams["font.family"] = "serif"
        except Exception:
            pass
