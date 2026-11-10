"""Run clean test-set retrieval ablations with bootstrap confidence intervals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_pipeline import HybridRetriever, evaluate, load_chunks, load_golden_set


META_CATEGORIES = {"source-selection", "conflict-resolution", "adversarial"}


def bootstrap_interval(values: list[float], seed: int = 2026) -> list[float]:
    rng = np.random.default_rng(seed)
    sample = rng.choice(np.asarray(values), size=(10000, len(values)), replace=True)
    means = sample.mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def metric_intervals(result: dict[str, object]) -> dict[str, list[float]]:
    details = result["details"]
    assert isinstance(details, list)
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    for row in details:
        assert isinstance(row, dict)
        rank = row["rank"]
        hits.append(float(rank is not None))
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
    return {
        "hit_at_5_ci95": bootstrap_interval(hits),
        "mrr_at_5_ci95": bootstrap_interval(reciprocal_ranks),
    }


def main() -> None:
    data_dir = Path("data/raw")
    source_gold = Path("Golden_set_300_rule_repaired.json")
    clean_gold = Path("Golden_set_300_clean.json")
    rows = load_golden_set(source_gold)
    clean_rows = [
        row for row in rows if row.get("category") not in META_CATEGORIES
    ]
    clean_gold.write_text(json.dumps(clean_rows, indent=2), encoding="utf-8")

    chunks = load_chunks(data_dir)
    retriever = HybridRetriever(chunks)
    common = {
        "data_dir": data_dir,
        "k": 5,
        "candidate_k": 120,
        "group_by": "page",
    }
    configurations = {
        "dense_only": {
            **common,
            "retrieval_mode": "dense",
            "bm25_weight": 0.0,
            "dense_weight": 1.0,
        },
        "bm25_only": {
            **common,
            "retrieval_mode": "bm25",
            "bm25_weight": 1.0,
            "dense_weight": 0.0,
        },
        "hybrid": {
            **common,
            "retrieval_mode": "hybrid",
            "bm25_weight": 0.3,
            "dense_weight": 0.7,
        },
        "hybrid_reranker": {
            **common,
            "retrieval_mode": "hybrid",
            "bm25_weight": 0.3,
            "dense_weight": 0.7,
            "rerank": True,
        },
    }
    results: dict[str, object] = {
        "test_count": sum(
            row.get("evaluation_split") == "test" and row.get("answerable", False)
            for row in clean_rows
        ),
        "dropped_meta_ids": [
            row.get("id")
            for row in rows
            if row.get("category") in META_CATEGORIES
        ],
        "gold_labeling": {
            "source": str(source_gold),
            "relevant_chunk_ids": (
                "Legacy chunk IDs are mapped to source/page targets using the "
                "original 1800-character, 300-overlap splitter."
            ),
            "cleaning": (
                "Dropped source-selection, conflict-resolution, and adversarial "
                "meta questions; retained multi-document questions."
            ),
        },
        "ablations": {},
    }
    for name, configuration in configurations.items():
        result = evaluate(
            retriever,
            clean_rows,
            split="test",
            **configuration,
        )
        results["ablations"][name] = {
            "configuration": {
                key: value for key, value in configuration.items() if key != "data_dir"
            },
            "count": result["count"],
            "hit_at_5": result["hit_at_k"],
            "mrr_at_5": result["mrr_at_k"],
            "hit_at_5_ci95": metric_intervals(result)["hit_at_5_ci95"],
            "mrr_at_5_ci95": metric_intervals(result)["mrr_at_5_ci95"],
        }
        print(
            name,
            f"Hit@5={result['hit_at_k']:.4f}",
            f"MRR@5={result['mrr_at_k']:.4f}",
            results["ablations"][name]["hit_at_5_ci95"],
            results["ablations"][name]["mrr_at_5_ci95"],
        )
    Path("ablation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
