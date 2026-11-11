"""Subsampled consensus NMF and module quality calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

from preprocessing import normalized_log_counts


@dataclass
class DiscoveryResult:
    W: np.ndarray
    H: np.ndarray
    stability: np.ndarray
    occurrence: np.ndarray
    reconstruction_error: float
    sparsity: np.ndarray
    top_genes: list[list[int]]
    matching_scores: np.ndarray
    k: int


def match_factors(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = reference / np.maximum(np.linalg.norm(reference, axis=0, keepdims=True), 1e-8)
    cand = candidate / np.maximum(np.linalg.norm(candidate, axis=0, keepdims=True), 1e-8)
    similarity = ref.T @ cand
    rows, cols = linear_sum_assignment(-similarity)
    order = np.empty(len(rows), dtype=np.int64)
    order[rows] = cols
    return order, similarity[rows, cols]


def _fit(matrix: np.ndarray, k: int, seed: int, max_iter: int):
    model = NMF(n_components=k, init="nndsvda", random_state=seed,
                max_iter=max_iter, solver="mu", beta_loss="kullback-leibler")
    h = model.fit_transform(matrix)
    w = model.components_.T
    scale = np.maximum(w.sum(axis=0), 1e-8)
    return w / scale, h * scale, float(model.reconstruction_err_)


def discover_modules(counts: np.ndarray, selected_genes: np.ndarray, k: int,
                     n_restarts: int = 50, cell_fraction: float = 0.8,
                     gene_fraction: float = 0.8, max_iter: int = 400,
                     random_state: int = 7, top_n: int = 50) -> DiscoveryResult:
    if k < 2 or n_restarts < 2:
        raise ValueError("k must be >= 2 and n_restarts must be >= 2")
    if counts.shape[0] < k or len(selected_genes) < k:
        raise ValueError("Both cell and selected-gene counts must be at least K")
    if not 0 < cell_fraction <= 1 or not 0 < gene_fraction <= 1:
        raise ValueError("subsampling fractions must be in (0, 1]")
    y = normalized_log_counts(counts)[:, selected_genes]
    n_cells, n_genes = y.shape
    runs: list[np.ndarray] = []
    errors: list[float] = []
    rng = np.random.default_rng(random_state)
    for restart in range(n_restarts):
        nc = max(k, int(round(n_cells * cell_fraction)))
        ng = max(k, int(round(n_genes * gene_fraction)))
        cells = rng.choice(n_cells, nc, replace=False)
        genes = np.sort(rng.choice(n_genes, ng, replace=False))
        w, _, error = _fit(y[np.ix_(cells, genes)], k, random_state + restart, max_iter)
        expanded = np.zeros((n_genes, k), dtype=np.float32)
        expanded[genes] = w
        runs.append(expanded)
        errors.append(error)
    ref = runs[int(np.argmin(errors))]
    aligned = []
    matching = []
    for run in runs:
        order, score = match_factors(ref, run)
        aligned.append(run[:, order])
        matching.append(score)
    consensus = np.mean(aligned, axis=0)
    consensus /= np.maximum(consensus.sum(axis=0, keepdims=True), 1e-8)
    stability = np.mean([
        np.sum(run * consensus, axis=0) /
        np.maximum(np.linalg.norm(run, axis=0) * np.linalg.norm(consensus, axis=0), 1e-8)
        for run in aligned
    ], axis=0).astype(np.float32)
    occurrence = np.mean([run.sum(axis=0) > 1e-8 for run in aligned], axis=0).astype(np.float32)

    # Fit H on all cells using the same consensus dictionary; module identities are fixed.
    final_w, final_h, error = _fit(y, k, random_state + n_restarts + 1, max_iter)
    order, _ = match_factors(consensus, final_w)
    final_w, final_h = final_w[:, order], final_h[:, order]
    final_w /= np.maximum(final_w.sum(axis=0, keepdims=True), 1e-8)
    effective = (consensus > (consensus.max(axis=0, keepdims=True) * 0.01)).sum(axis=0)
    top_genes = [np.argsort(consensus[:, i])[::-1][:top_n].tolist() for i in range(k)]
    return DiscoveryResult(consensus.astype(np.float32), final_h.astype(np.float32),
                           stability, occurrence, error, effective.astype(np.int32),
                           top_genes, np.asarray(matching, dtype=np.float32), k)


def module_redundancy(w: np.ndarray, top_n: int = 50, cosine_threshold: float = 0.8,
                      jaccard_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    k = w.shape[1]
    norm = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-8)
    similarity = norm.T @ norm
    tops = [set(np.argsort(w[:, i])[::-1][:top_n]) for i in range(k)]
    parent = list(range(k))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(k):
        for j in range(i):
            jac = len(tops[i] & tops[j]) / max(1, len(tops[i] | tops[j]))
            if similarity[i, j] >= cosine_threshold and jac >= jaccard_threshold:
                parent[find(i)] = find(j)
    clusters = np.asarray([find(i) for i in range(k)])
    _, inverse = np.unique(clusters, return_inverse=True)
    return inverse, similarity.astype(np.float32)


def merge_redundant_modules(result: DiscoveryResult, top_n: int = 50,
                            cosine_threshold: float = 0.8,
                            jaccard_threshold: float = 0.5) -> tuple[DiscoveryResult, list[list[int]], np.ndarray]:
    cluster, similarity = module_redundancy(result.W, top_n, cosine_threshold, jaccard_threshold)
    n_new = int(cluster.max()) + 1
    w = np.zeros((result.W.shape[0], n_new), dtype=np.float32)
    h = np.zeros((result.H.shape[0], n_new), dtype=np.float32)
    merged_from = []
    for new_id in range(n_new):
        old = np.flatnonzero(cluster == new_id).tolist()
        merged_from.append(old)
        w[:, new_id] = result.W[:, old].mean(axis=1)
        h[:, new_id] = result.H[:, old].sum(axis=1)
    w /= np.maximum(w.sum(axis=0, keepdims=True), 1e-8)
    top = [np.argsort(w[:, i])[::-1][:top_n].tolist() for i in range(n_new)]
    merged = DiscoveryResult(w, h, np.asarray([result.stability[x].max() for x in
                                                [np.flatnonzero(cluster == i) for i in range(n_new)]]),
                             np.asarray([result.occurrence[x].mean() for x in
                                         [np.flatnonzero(cluster == i) for i in range(n_new)]]),
                             result.reconstruction_error,
                             np.asarray([(w[:, i] > w[:, i].max() * .01).sum() for i in range(n_new)]),
                             top, result.matching_scores, n_new)
    return merged, merged_from, similarity
