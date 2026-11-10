"""Audit and repair every gold mapping using document and non-TOC rules."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import fitz

from rag_pipeline import _legacy_page_targets, is_table_of_contents, tokenize


def page_scores(
    data_dir: Path, source: str, question: str
) -> list[tuple[int, int]]:
    terms = [term for term in tokenize(question) if len(term) > 2]
    with fitz.open(data_dir / source) as document:
        scored = []
        for page_number, page in enumerate(document, start=1):
            text = page.get_text()
            if is_table_of_contents(text):
                continue
            lower = text.lower()
            score = sum(lower.count(term) for term in terms)
            if score:
                scored.append((score, page_number))
    return sorted(scored, key=lambda item: (-item[0], item[1]))


def main() -> None:
    data_dir = Path("data/raw")
    rows = json.loads(Path("Golden_set_300.json").read_text(encoding="utf-8"))
    legacy_targets = _legacy_page_targets(data_dir)
    toc_pages: dict[str, set[int]] = {}
    for source in {path.name for path in data_dir.glob("*.pdf")}:
        with fitz.open(source if Path(source).is_absolute() else data_dir / source) as document:
            toc_pages[source] = {
                page_number
                for page_number, page in enumerate(document, start=1)
                if is_table_of_contents(page.get_text())
            }

    issue_rows = []
    issue_types = Counter()
    family_issues = Counter()
    for row in rows:
        if not row.get("answerable"):
            continue
        allowed = {row.get("doc_id")} | set(row.get("supporting_doc_ids") or [])
        old_pages = {
            legacy_targets[chunk_id]
            for chunk_id in row.get("relevant_chunk_ids", [])
            if chunk_id in legacy_targets
        }
        valid_pages = [
            page for page in old_pages
            if page[0] in allowed and page[1] not in toc_pages[page[0]]
        ]
        issues = []
        if not old_pages:
            issues.append("no_legacy_target")
        if any(source not in allowed for source, _ in old_pages):
            issues.append("outside_allowed_docs")
        if any(page in toc_pages[source] for source, page in old_pages):
            issues.append("toc")
        if issues:
            issue_rows.append(row)
            issue_types.update(issues)
            family_issues[row.get("doc_id")] += 1
        if valid_pages:
            row["relevant_pages"] = [
                {"source": source, "page": page}
                for source, page in sorted(valid_pages)
            ]
        else:
            candidates = []
            for source in sorted(allowed - {None}):
                candidates.extend(
                    (score, source, page)
                    for score, page in page_scores(data_dir, source, row["question"])
                )
            if not candidates:
                raise ValueError(f"No non-TOC evidence page found for {row.get('id')}")
            _, source, page = max(candidates, key=lambda item: (item[0], -item[2]))
            row["relevant_pages"] = [{"source": source, "page": page}]
            row["label_mapping_note"] = (
                "Automatically repaired: target is restricted to doc_id/supporting "
                "docs and selected from a non-TOC page by lexical evidence."
            )

    output = Path("Golden_set_300_rule_repaired.json")
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = {
        "answerable_rows": sum(row.get("answerable", False) for row in rows),
        "mismatch_rows": len(issue_rows),
        "issue_types": dict(issue_types),
        "mismatch_by_doc_id": dict(family_issues),
        "mismatch_ids": [row.get("id") for row in issue_rows],
        "rule": (
            "Gold pages must be in doc_id/supporting_doc_ids and must not be "
            "automatically detected TOC pages; TOC-only rows use lexical evidence "
            "over non-TOC pages in allowed documents."
        ),
    }
    Path("gold_mapping_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
