"""Select embedding/chunking/reranker variants on development questions only."""

from __future__ import annotations

import json
from pathlib import Path

from rag_pipeline import HybridRetriever, evaluate, load_chunks, load_golden_set


META_CATEGORIES = {"source-selection", "conflict-resolution", "adversarial"}


def main() -> None:
    data_dir = Path("data/raw")
    rows = [
        row
        for row in load_golden_set("Golden_set_300_rule_repaired.json")
        if row.get("category") not in META_CATEGORIES
    ]
    variants = [
        {
            "name": "minilm_1400",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 1400,
            "chunk_overlap": 180,
            "include_section_heading": False,
        },
        {
            "name": "bge_base_1000_heading",
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "chunk_size": 1000,
            "chunk_overlap": 120,
            "include_section_heading": True,
        },
    ]
    results = []
    output_path = Path("model_experiment_results.json")
    for variant in variants:
        chunks = load_chunks(
            data_dir,
            chunk_size=variant["chunk_size"],
            chunk_overlap=variant["chunk_overlap"],
            include_section_heading=variant["include_section_heading"],
        )
        retriever = HybridRetriever(
            chunks, embedding_model=variant["embedding_model"]
        )
        result = evaluate(
            retriever,
            rows,
            split="development",
            data_dir=data_dir,
            k=5,
            candidate_k=120,
            bm25_weight=0.3,
            dense_weight=0.7,
            group_by="page",
        )
        results.append(
            {
                **variant,
                "chunk_count": len(chunks),
                "hit_at_5": result["hit_at_k"],
                "mrr_at_5": result["mrr_at_k"],
                "audited_hit_at_5": result["audited_hit_at_k"],
                "audited_mrr_at_5": result["audited_mrr_at_k"],
            }
        )
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(results[-1])


if __name__ == "__main__":
    main()
