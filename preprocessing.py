"""Input validation, cell QC and biology-guided gene selection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def load_input(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        data = {key: np.asarray(value) for key, value in archive.items()}
    if "counts" not in data:
        raise ValueError("Input .npz must contain a 'counts' array")
    counts = data["counts"]
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite, non-negative cells x genes matrix")
    n_cells = counts.shape[0]
    for key in ("is_control", "perturbation", "context", "batch"):
        if key in data and data[key].ndim != 1 or key in data and len(data[key]) != n_cells:
            raise ValueError(f"{key} must have one value per cell")
    return data


def qc_mask(counts: np.ndarray, min_counts: float = 0.0,
            min_genes: int = 0, max_mito_fraction: float | None = None,
            gene_names: Iterable[str] | None = None) -> tuple[np.ndarray, dict]:
    """Return a cell mask and an auditable QC summary.

    QC thresholds default to zero so existing datasets are not silently altered.
    """
    library = counts.sum(axis=1)
    detected = (counts > 0).sum(axis=1)
    mask = (library >= min_counts) & (detected >= min_genes)
    mito_fraction = None
    if max_mito_fraction is not None:
        if gene_names is None:
            raise ValueError("max_mito_fraction requires gene names")
        names = np.asarray(list(gene_names), dtype=str)
        mito = np.char.startswith(np.char.upper(names), "MT-")
        if not np.any(mito):
            raise ValueError("No MT- genes found for mitochondrial QC")
        mito_fraction = counts[:, mito].sum(axis=1) / np.maximum(library, 1.0)
        mask &= mito_fraction <= max_mito_fraction
    report = {
        "input_cells": int(len(counts)),
        "retained_cells": int(mask.sum()),
        "removed_cells": int((~mask).sum()),
        "min_counts": min_counts,
        "min_genes": min_genes,
        "max_mito_fraction": max_mito_fraction,
    }
    if mito_fraction is not None:
        report["median_mito_fraction"] = float(np.median(mito_fraction))
    if not np.any(mask):
        raise ValueError("QC removed all cells")
    return mask, report


def read_gene_names(path: str | None, n_genes: int) -> np.ndarray:
    if path is None:
        return np.asarray([f"gene_{i}" for i in range(n_genes)], dtype=str)
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(names) != n_genes:
        raise ValueError(f"{path} contains {len(names)} genes, expected {n_genes}")
    return np.asarray(names, dtype=str)


def read_gene_sets(paths: Iterable[str] | None) -> set[str]:
    """Read one-gene-per-line or GMT files and return their gene union."""
    result: set[str] = set()
    for raw_path in paths or []:
        for line in Path(raw_path).read_text(encoding="utf-8").splitlines():
            fields = line.strip().split("\t")
            if not fields or not fields[0] or fields[0].startswith("#"):
                continue
            if len(fields) >= 3:  # GMT: set name, description, genes...
                result.update(g for g in fields[2:] if g)
            else:
                result.add(fields[0])
    return result


def normalized_log_counts(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    library = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(counts / library * target_sum).astype(np.float32)


def select_genes(counts: np.ndarray, gene_names: np.ndarray,
                 min_genes: int = 4000, max_genes: int = 8000,
                 guided_genes: set[str] | None = None,
                 min_cells: int = 3, row_indices: np.ndarray | None = None,
                 chunk_size: int = 512) -> tuple[np.ndarray, dict]:
    """Select HVGs and union them with supplied biology-guided genes."""
    if min_genes < 1 or max_genes < min_genes:
        raise ValueError("Require 1 <= min_genes <= max_genes")
    rows = np.arange(len(counts)) if row_indices is None else np.asarray(row_indices)
    n = len(rows)
    # Stream rows so a dense float64 cells-by-genes matrix is never duplicated.
    total = np.zeros(counts.shape[1], dtype=np.float64)
    total_sq = np.zeros(counts.shape[1], dtype=np.float64)
    expressed_count = np.zeros(counts.shape[1], dtype=np.int64)
    for start in range(0, n, chunk_size):
        chunk = np.asarray(counts[rows[start:start + chunk_size]], dtype=np.float32)
        y_chunk = normalized_log_counts(chunk)
        total += y_chunk.sum(axis=0, dtype=np.float64)
        total_sq += np.square(y_chunk, dtype=np.float32).sum(axis=0, dtype=np.float64)
        expressed_count += (chunk > 0).sum(axis=0)
    variance = np.maximum(total_sq / max(n, 1) - np.square(total / max(n, 1)), 0.0)
    expressed = expressed_count >= min_cells
    target = min(counts.shape[1], max_genes)
    hvg_n = min(target, max(min_genes, int(target * 0.8)))
    hvg = np.argsort(variance)[-hvg_n:]
    guided_mask = np.isin(gene_names, list(guided_genes or set())) & expressed
    selected = np.unique(np.concatenate([hvg, np.flatnonzero(guided_mask)]))
    if len(selected) > target:
        # Keep every guided gene, then fill remaining slots by variance.
        guided_idx = np.flatnonzero(guided_mask)
        remaining = np.setdiff1d(np.argsort(variance)[::-1], guided_idx,
                                 assume_unique=False)
        selected = np.concatenate([guided_idx, remaining[:target - len(guided_idx)]])
    selected = np.sort(selected)
    if len(selected) < min_genes:
        selected = np.argsort(variance)[-min(counts.shape[1], min_genes):]
    return selected.astype(np.int64), {
        "input_genes": int(counts.shape[1]),
        "selected_genes": int(len(selected)),
        "hvg_genes": int(len(hvg)),
        "guided_genes_present": int(guided_mask.sum()),
        "min_cells": min_cells,
    }
