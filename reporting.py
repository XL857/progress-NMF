"""Small, optional visual reports for module discovery."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_plots(output_dir: str, w: np.ndarray, h: np.ndarray, stability: np.ndarray,
               k_scores: dict[int, float] | None = None) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(stability)), stability, "o")
    plt.xlabel("module")
    plt.ylabel("stability")
    plt.tight_layout()
    plt.savefig(path / "module_stability.png", dpi=150)
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.imshow(w, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.xlabel("module")
    plt.ylabel("selected gene")
    plt.colorbar(label="loading")
    plt.tight_layout()
    plt.savefig(path / "loading_heatmap.png", dpi=150)
    plt.close()
    if k_scores:
        plt.figure(figsize=(6, 4))
        ks = sorted(k_scores)
        plt.plot(ks, [k_scores[k] for k in ks], "o-")
        plt.xlabel("K")
        plt.ylabel("selection score")
        plt.tight_layout()
        plt.savefig(path / "k_selection.png", dpi=150)
        plt.close()
