"""Optional gene-set enrichment for module reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import hypergeom


def read_gmt(path: str) -> dict[str, set[str]]:
    sets = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) >= 3:
            sets[fields[0]] = set(fields[2:])
    return sets


def enrich_modules(top_genes: list[list[str]], gene_set_files: list[str] | None,
                   background: set[str] | None = None, top_terms: int = 10) -> list[list[dict]]:
    gene_sets = {}
    for path in gene_set_files or []:
        gene_sets.update(read_gmt(path))
    bg = background or {gene for genes in top_genes for gene in genes}
    output = []
    for genes in top_genes:
        query = set(genes) & bg
        rows = []
        for name, members in gene_sets.items():
            overlap = len(query & members & bg)
            if overlap == 0:
                continue
            p = hypergeom.sf(overlap - 1, len(bg), len(members & bg), len(query))
            rows.append({"term": name, "overlap": overlap, "p_value": float(p)})
        rows.sort(key=lambda row: (row["p_value"], -row["overlap"]))
        output.append(rows[:top_terms])
    return output
