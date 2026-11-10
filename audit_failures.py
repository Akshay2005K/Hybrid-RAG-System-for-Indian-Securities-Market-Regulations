"""Classify clean-test misses using the saved page-level evaluation output."""

from __future__ import annotations

import json
from pathlib import Path


META_CATEGORIES = {"source-selection", "conflict-resolution", "adversarial"}


def main() -> None:
    gold = json.loads(Path("Golden_set_300.json").read_text(encoding="utf-8"))
    clean_ids = {
        row["id"]
        for row in gold
        if row.get("evaluation_split") == "test"
        and row.get("category") not in META_CATEGORIES
    }
    results = json.loads(
        Path("evaluation_results_chunk_optimized.json").read_text(encoding="utf-8")
    )
    test_results = results["test"]
    selected = (
        max(test_results, key=lambda result: result.get("audited_mrr_at_k", -1))
        if isinstance(test_results, list)
        else test_results
    )
    failures = []
    for row in selected["details"]:
        if row["id"] not in clean_ids or row["rank"] is not None:
            continue
        target_sources = {target[0] for target in row["content_target_pages"]}
        retrieved_sources = {item["source"] for item in row["retrieved"]}
        target_pages = {tuple(target) for target in row["content_target_pages"]}
        retrieved_pages = {
            (item["source"], item["page"]) for item in row["retrieved"]
        }
        if target_sources & retrieved_sources:
            category = "right_document_wrong_page"
        else:
            category = "wrong_document"
        failures.append(
            {
                "id": row["id"],
                "question": row["question"],
                "category": category,
                "target_pages": row["content_target_pages"],
                "retrieved": row["retrieved"],
                "same_page_in_top5": bool(target_pages & retrieved_pages),
            }
        )
    summary = {
        "source_evaluation": "evaluation_results_chunk_optimized.json",
        "clean_test_count": len(clean_ids),
        "failure_count": len(failures),
        "right_document_wrong_page": sum(
            item["category"] == "right_document_wrong_page" for item in failures
        ),
        "wrong_document": sum(
            item["category"] == "wrong_document" for item in failures
        ),
        "failures": failures,
    }
    Path("failure_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "failures"}, indent=2))


if __name__ == "__main__":
    main()
