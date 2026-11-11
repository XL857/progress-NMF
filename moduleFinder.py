"""Single-cell feature-module discovery pipeline.

The script is intentionally an orchestration layer. Heavy functionality lives in
preprocessing.py, cnmf_discovery.py, annotations.py and reporting.py.

Example:
    python moduleFinder.py --input baseline.npz --output modules.npz \
        --k-grid 64,96,128 --restarts 50 --genes genes.txt \
        --gene-set hallmarks.gmt --report-dir module_report
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from annotations import enrich_modules
from cnmf_discovery import (DiscoveryResult, discover_modules,
                            match_factors, merge_redundant_modules)
from preprocessing import (load_input, qc_mask, read_gene_names, read_gene_sets,
                           select_genes)
from reporting import save_plots


def parse_grid(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("k-grid must contain integers >= 2")
    return sorted(set(values))


def select_basal(data: dict[str, np.ndarray]) -> np.ndarray:
    n = len(data["counts"])
    if "is_control" in data:
        mask = data["is_control"].astype(bool)
    elif "perturbation" in data:
        mask = data["perturbation"] == 0
    else:
        mask = np.ones(n, dtype=bool)
    if mask.ndim != 1 or len(mask) != n or not np.any(mask):
        raise ValueError("No valid control/basal cells were found")
    return mask


def filter_modules(result: DiscoveryResult, stability_threshold: float,
                   occurrence_threshold: float, min_effective_genes: int) -> tuple[DiscoveryResult, np.ndarray]:
    keep = ((result.stability >= stability_threshold) &
            (result.occurrence >= occurrence_threshold) &
            (result.sparsity >= min_effective_genes))
    if not np.any(keep):
        # Preserve the best module rather than producing an unusable dictionary.
        keep[np.argmax(result.stability)] = True
    filtered = DiscoveryResult(result.W[:, keep], result.H[:, keep], result.stability[keep],
                                result.occurrence[keep], result.reconstruction_error,
                                result.sparsity[keep],
                                [genes for genes, flag in zip(result.top_genes, keep) if flag],
                                result.matching_scores, int(keep.sum()))
    return filtered, keep


def align_contexts(global_w: np.ndarray, context_results: dict[str, DiscoveryResult],
                   similarity_threshold: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Align per-context dictionaries to global modules and mark presence."""
    n_context = len(context_results)
    n_genes, n_modules = global_w.shape
    context_w = np.zeros((n_context, n_genes, n_modules), dtype=np.float32)
    presence = np.zeros((n_context, n_modules), dtype=np.int8)
    metadata = {}
    for context_id, (label, result) in enumerate(context_results.items()):
        order, scores = match_factors(global_w, result.W)
        for global_id, local_id in enumerate(order):
            score = float(scores[global_id])
            if score >= similarity_threshold:
                context_w[context_id, :, global_id] = result.W[:, local_id]
                presence[context_id, global_id] = 1
        metadata[str(label)] = {
            "matched_similarity": scores.tolist(),
            "context_specific_modules": [int(i) for i, score in enumerate(scores)
                                          if score < similarity_threshold],
        }
    return context_w, presence, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k-grid", type=parse_grid, default=parse_grid("32,48,64,80,96,128,160"))
    parser.add_argument("--restarts", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--cell-fraction", type=float, default=0.8)
    parser.add_argument("--gene-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--genes", help="gene names, one per line")
    parser.add_argument("--gene-set", action="append", default=[],
                        help="GMT or one-gene-per-line file; may be repeated")
    parser.add_argument("--min-counts", type=float, default=0.0)
    parser.add_argument("--min-cell-genes", type=int, default=0)
    parser.add_argument("--max-mito-fraction", type=float)
    parser.add_argument("--min-genes", type=int, default=4000)
    parser.add_argument("--max-genes", type=int, default=8000)
    parser.add_argument("--min-expressed-cells", type=int, default=3)
    parser.add_argument("--stability-threshold", type=float, default=0.85)
    parser.add_argument("--occurrence-threshold", type=float, default=0.5)
    parser.add_argument("--min-effective-genes", type=int, default=20)
    parser.add_argument("--redundancy-cosine", type=float, default=0.8)
    parser.add_argument("--redundancy-jaccard", type=float, default=0.5)
    parser.add_argument("--context-threshold", type=float, default=0.6)
    parser.add_argument("--context-key", default="context",
                        help="metadata field for per-context discovery")
    parser.add_argument("--no-context-discovery", action="store_true")
    parser.add_argument("--report-dir")
    args = parser.parse_args()

    data = load_input(args.input)
    n_input_genes = data["counts"].shape[1]
    genes = read_gene_names(args.genes, n_input_genes)
    mask, qc = qc_mask(data["counts"], args.min_counts, args.min_cell_genes,
                       args.max_mito_fraction, genes)
    counts = data["counts"]
    basal = select_basal(data) & mask
    basal_indices = np.flatnonzero(basal)
    guided = read_gene_sets(args.gene_set)
    selected, gene_report = select_genes(counts, genes, args.min_genes,
                                         args.max_genes, guided, args.min_expressed_cells,
                                         row_indices=basal_indices)
    selected_local = np.arange(len(selected), dtype=np.int64)
    # Only this reduced matrix is needed after feature selection.
    basal_counts = np.asarray(counts[np.ix_(basal_indices, selected)], dtype=np.float32)
    del data["counts"]
    del counts
    gc.collect()

    results: dict[int, DiscoveryResult] = {}
    scores: dict[int, float] = {}
    score_details = {}
    for k in args.k_grid:
        if k > min(basal_counts.shape[0], len(selected)):
            continue
        result = discover_modules(basal_counts, selected_local, k, args.restarts,
                                  args.cell_fraction, args.gene_fraction,
                                  args.max_iter, args.seed + k)
        # Redundancy is reported before merging and is part of K selection.
        _, _, similarity = merge_redundant_modules(result, cosine_threshold=args.redundancy_cosine,
                                                   jaccard_threshold=args.redundancy_jaccard)
        upper = similarity[np.triu_indices(k, 1)]
        redundancy = float(np.mean(upper >= args.redundancy_cosine)) if len(upper) else 0.0
        score = float(result.stability.mean() - 0.25 * redundancy)
        results[k], scores[k] = result, score
        score_details[k] = {"stability": float(result.stability.mean()),
                            "redundancy": redundancy, "downstream_performance": None,
                            "annotation_coherence": None, "score": score}
    if not results:
        raise ValueError("No K in k-grid is feasible for the available cells and genes")
    best_k = max(scores, key=scores.get)
    result, keep = filter_modules(results[best_k], args.stability_threshold,
                                  args.occurrence_threshold, args.min_effective_genes)
    result, merged_from, similarity = merge_redundant_modules(
        result, cosine_threshold=args.redundancy_cosine,
        jaccard_threshold=args.redundancy_jaccard)

    full_w = np.zeros((n_input_genes, result.W.shape[1]), dtype=np.float32)
    full_w[selected] = result.W
    top_names = [[str(genes[selected[i]]) for i in module] for module in result.top_genes]
    enrichment = enrich_modules(top_names, args.gene_set, set(genes[selected]))

    context_results = {}
    context_w = np.empty((0, full_w.shape[0], full_w.shape[1]), dtype=np.float32)
    context_presence = np.empty((0, full_w.shape[1]), dtype=np.int8)
    context_metadata = {}
    if (not args.no_context_discovery and args.context_key in data):
        labels = data[args.context_key][basal]
        for label in np.unique(labels):
            context_counts = basal_counts[labels == label]
            if len(context_counts) >= best_k:
                context_results[str(label)] = discover_modules(
                    context_counts, selected_local, best_k, args.restarts,
                    args.cell_fraction, args.gene_fraction, args.max_iter, args.seed + best_k)
        if context_results:
            aligned_inputs = {}
            for label, value in context_results.items():
                context_full_w = np.zeros_like(full_w)
                context_full_w[selected] = value.W
                aligned_inputs[label] = DiscoveryResult(
                    context_full_w, value.H, value.stability, value.occurrence,
                    value.reconstruction_error, value.sparsity, value.top_genes,
                    value.matching_scores, value.k)
            context_w, context_presence, context_metadata = align_contexts(
                full_w, aligned_inputs, args.context_threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, W=full_w, H=result.H, stability=result.stability,
                        occurrence=result.occurrence, effective_genes=result.sparsity,
                        selected_genes=selected, genes=genes, context_W=context_w,
                        context_presence=context_presence, k_grid=np.asarray(sorted(results)),
                        k_scores=np.asarray([scores[k] for k in sorted(results)]))
    metadata = {
        "input": str(args.input), "basal_cells": int(basal.sum()), "qc": qc,
        "gene_selection": gene_report, "selected_gene_names": genes[selected].tolist(),
        "chosen_k_before_filtering": int(best_k), "modules_after_filter_and_merge": int(result.k),
        "quality_thresholds": {"stability": args.stability_threshold,
                               "occurrence": args.occurrence_threshold,
                               "min_effective_genes": args.min_effective_genes},
        "module_metadata": [{"module_id": f"M{i:03d}", "top_genes": top_names[i],
                             "stability": float(result.stability[i]),
                             "bootstrap_occurrence": float(result.occurrence[i]),
                             "effective_genes": int(result.sparsity[i]),
                             "merged_from": merged_from[i], "enrichment": enrichment[i]}
                            for i in range(result.k)],
        "k_selection": score_details, "context": context_metadata,
        "matching_score_mean": float(result.matching_scores.mean()),
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report_dir:
        save_plots(args.report_dir, result.W, result.H, result.stability, scores)
    print(json.dumps({"output": str(output), "metadata": str(metadata_path),
                      "basal_cells": int(basal.sum()), "selected_genes": int(len(selected)),
                      "chosen_k": best_k, "final_modules": result.k,
                      "mean_stability": float(result.stability.mean())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
