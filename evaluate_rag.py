"""Tune retrieval on development questions and score the test split once.

Usage:
    python evaluate_rag.py --data-dir data/raw --gold Golden_set_300.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_pipeline import HybridRetriever, evaluate, load_chunks, load_golden_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--gold", type=Path, default=Path("Golden_set_300.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    args = parser.parse_args()

    chunks = load_chunks(args.data_dir)
    retriever = HybridRetriever(chunks)
    golden = load_golden_set(args.gold)
    search_space = [
        {
            "bm25_weight": weight,
            "dense_weight": 1.0 - weight,
            "candidate_k": candidate_k,
            "group_by": group_by,
        }
        for weight in (0.1, 0.2, 0.3, 0.35, 0.4)
        for candidate_k in (80, 120)
        for group_by in ("page", "source")
    ]
    development = [
        evaluate(
            retriever,
            golden,
            split="development",
            data_dir=args.data_dir,
            k=5,
            **configuration,
        )
        | {"configuration": configuration}
        for configuration in search_space
    ]
    best = max(
        development,
        key=lambda result: (
            result["audited_mrr_at_k"],
            result["audited_hit_at_k"],
            result["mrr_at_k"],
        ),
    )
    test = evaluate(
        retriever,
        golden,
        split="test",
        data_dir=args.data_dir,
        k=5,
        **best["configuration"],
    )
    result = {
        "chunk_count": len(chunks),
        "selected_configuration": best["configuration"],
        "development": development,
        "test": test,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"development Hit@5={best['hit_at_k']:.4f} "
        f"MRR@5={best['mrr_at_k']:.4f} "
        f"SourceHit@5={best['source_hit_at_k']:.4f} "
        f"SourceMRR@5={best['source_mrr_at_k']:.4f}"
    )
    print(
        f"test Hit@5={test['hit_at_k']:.4f} MRR@5={test['mrr_at_k']:.4f} "
        f"SourceHit@5={test['source_hit_at_k']:.4f} "
        f"SourceMRR@5={test['source_mrr_at_k']:.4f}"
    )
    print(f"Detailed results written to {args.output}")


if __name__ == "__main__":
    main()
