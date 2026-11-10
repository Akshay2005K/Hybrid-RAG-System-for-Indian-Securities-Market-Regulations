"""Leakage-safe hybrid retrieval for the stock-market PDF corpus.

The module keeps ingestion, indexing, retrieval, and evaluation independent of
notebook state. It does not train on evaluation questions or tune on the test
split.
"""

from __future__ import annotations

import csv
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import fitz
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


__all__ = [
    "Chunk",
    "HybridRetriever",
    "evaluate",
    "extractive_answer",
    "has_strong_match",
    "is_table_of_contents",
    "load_chunks",
    "load_golden_set",
    "load_manifest",
    "load_sentence_chunks",
    "normalise_page_text",
    "tokenize",
    "generate_ollama_answer",
    "generate_groq_answer",
]

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[+#-][a-z0-9]+)*", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ANSWER_STOPWORDS = {
    "about", "after", "also", "and", "are", "does", "for", "from", "how",
    "into", "is", "it", "of", "on", "or", "that", "the", "their", "these",
    "this", "to", "what", "when", "where", "which", "who", "why", "with",
}


def tokenize(text: str) -> list[str]:
    """Tokenize terms such as T+1, UPI, NAV, ASBA, and numerical values."""
    return TOKEN_RE.findall(text.lower())


def load_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    """Load document metadata keyed by local PDF filename."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return {
            str(row["local_filename"]): {
                "title": str(row.get("title", "")),
                "publisher": str(row.get("publisher", "")),
                "doc_date": str(row.get("doc_date", "")),
                "source_url": str(row.get("source_url", "")),
                "status": (
                    "superseded"
                    if "superseded" in str(row.get("notes", "")).lower()
                    else "current"
                ),
            }
            for row in csv.DictReader(handle)
            if row.get("local_filename")
        }


def has_strong_match(
    query: str,
    results: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.45,
) -> bool:
    """Require sufficient query-term coverage before showing an answer."""
    query_terms = {
        term
        for term in tokenize(query)
        if term not in ANSWER_STOPWORDS and len(term) > 2
    }
    if not query_terms:
        return False
    best_coverage = max(
        (
            len(query_terms.intersection(tokenize(str(result.get("text", ""))))
                ) / len(query_terms)
            for result in results
        ),
        default=0.0,
    )
    return best_coverage >= threshold


def extractive_answer(
    query: str,
    results: Iterable[dict[str, Any]],
    *,
    max_sentences: int = 3,
) -> dict[str, Any]:
    """Select answer-bearing sentences from retrieved evidence.

    This is deliberately extractive: every returned sentence comes from a
    retrieved chunk, so the demo can answer without an external generation
    service or unsupported claims.
    """
    query_terms = {
        term for term in tokenize(query) if term not in ANSWER_STOPWORDS
    }
    candidates: list[tuple[float, int, str, dict[str, Any]]] = []
    for evidence_rank, result in enumerate(results, start=1):
        sentences = [
            sentence.strip()
            for sentence in SENTENCE_RE.split(str(result.get("text", "")))
            if len(sentence.strip().split()) >= 5
        ]
        for sentence in sentences:
            sentence_terms = set(tokenize(sentence))
            overlap = query_terms.intersection(sentence_terms)
            if not overlap:
                continue
            score = (
                len(overlap) / max(len(query_terms), 1)
                + 0.15 / evidence_rank
                + min(len(sentence.split()), 45) / 450
            )
            candidates.append((score, evidence_rank, sentence, result))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_terms: set[str] = set()
    for _, _, sentence, result in candidates:
        terms = set(tokenize(sentence))
        if terms and len(terms.intersection(selected_terms)) / len(terms) > 0.75:
            continue
        selected.append((sentence, result))
        selected_terms.update(terms)
        if len(selected) == max_sentences:
            break
    citations = [
        {
            "source": result["source"],
            "page": result["page"],
            "chunk_id": result["chunk_id"],
        }
        for _, result in selected
    ]
    return {
        "answer": " ".join(sentence for sentence, _ in selected),
        "citations": citations,
        "grounded": bool(selected),
    }


def normalise_page_text(text: str) -> str:
    """Remove extraction noise while preserving line and paragraph boundaries."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_table_of_contents(page_text: str) -> bool:
    text = normalise_page_text(page_text)
    if not text:
        return False
    lines = text.splitlines()
    has_header = any(
        re.search(r"^(table of contents|contents)\b", line.lower())
        for line in lines[:6]
    )
    dotted_lines = sum(
        "...." in line or re.search(r"\.{3,}\s*\d+$", line) is not None
        for line in lines
    )
    numbered_lines = sum(
        re.search(r"\s+\d+$", line.strip()) is not None for line in lines
    )
    return (has_header and dotted_lines >= 1) or (
        len(lines) > 5 and numbered_lines / len(lines) > 0.4 and dotted_lines >= 2
    )


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    text: str
    source: str
    page: int
    section: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "section": self.section,
        }


def _section_for_page(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if 3 <= len(candidate) <= 120 and not candidate.endswith((".", ":", ";")):
            if len(candidate.split()) <= 14:
                return candidate
    return ""


def load_chunks(
    data_dir: str | Path,
    *,
    chunk_size: int = 1400,
    chunk_overlap: int = 180,
    include_section_heading: bool = False,
) -> list[Chunk]:
    """Extract cleaned pages and split each page with source/page provenance."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for pdf_path in sorted(Path(data_dir).glob("*.pdf")):
        with fitz.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                raw_text = page.get_text()
                if is_table_of_contents(raw_text):
                    continue
                text = normalise_page_text(raw_text)
                if not text:
                    continue
                section = _section_for_page(text)
                for split in splitter.split_text(text):
                    indexed_text = (
                        f"{section}\n{split}" if include_section_heading and section
                        else split
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=len(chunks),
                            text=indexed_text,
                            source=pdf_path.name,
                            page=page_number,
                            section=section,
                        )
                    )
    if not chunks:
        raise ValueError(f"No usable PDF text found in {Path(data_dir)}")
    return chunks


def load_sentence_chunks(data_dir: str | Path) -> list[Chunk]:
    """Extract one searchable record per sentence while retaining page metadata."""
    sentence_chunks: list[Chunk] = []
    for pdf_path in sorted(Path(data_dir).glob("*.pdf")):
        with fitz.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                raw_text = page.get_text()
                if is_table_of_contents(raw_text):
                    continue
                text = normalise_page_text(raw_text)
                if not text:
                    continue
                section = _section_for_page(text)
                sentences = [
                    sentence.strip()
                    for sentence in SENTENCE_RE.split(text)
                    if len(sentence.strip().split()) >= 5
                ]
                for sentence in sentences:
                    sentence_chunks.append(
                        Chunk(
                            chunk_id=len(sentence_chunks),
                            text=sentence,
                            source=pdf_path.name,
                            page=page_number,
                            section=section,
                        )
                    )
    if not sentence_chunks:
        raise ValueError(f"No usable PDF sentences found in {Path(data_dir)}")
    return sentence_chunks


def generate_ollama_answer(
    query: str,
    results: Iterable[dict[str, Any]],
    *,
    model: str = "llama3.2:3b",
    endpoint: str = "http://localhost:11434/api/generate",
    timeout: int = 120,
) -> str:
    """Generate an answer from retrieved evidence using a local Ollama model."""
    evidence = "\n".join(
        f"[{index}] {result['text']}"
        for index, result in enumerate(results, start=1)
    )
    if not evidence:
        raise ValueError("Cannot generate an answer without retrieved evidence")
    prompt = (
        "Answer the question using only the supplied evidence. "
        "Do not use outside knowledge, do not invent facts, and say "
        "'The documents do not provide enough information.' if the evidence "
        "does not answer the question. Give a concise answer.\n\n"
        f"Question: {query}\n\nEvidence:\n{evidence}"
    )
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama is unavailable. Start Ollama and pull the selected model."
        ) from error
    answer = str(body.get("response", "")).strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer")
    return answer


def generate_groq_answer(
    query: str,
    results: Iterable[dict[str, Any]],
    *,
    model: str = "openai/gpt-oss-20b",
) -> str:
    """Generate a grounded answer from retrieved evidence using Groq."""
    evidence = "\n".join(
        f"[{index}] {result['text']}"
        for index, result in enumerate(results, start=1)
    )
    if not evidence:
        raise ValueError("Cannot generate an answer without retrieved evidence")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Configure it before starting Streamlit."
        )
    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError(
            "The Groq package is not installed. Run: pip install groq"
        ) from error
    prompt = (
        "Answer the question using only the supplied evidence. "
        "Do not use outside knowledge, do not invent facts, and say "
        "'The documents do not provide enough information.' if the evidence "
        "does not answer the question. Give a concise answer.\n\n"
        f"Question: {query}\n\nEvidence:\n{evidence}"
    )
    try:
        response = Groq(api_key=api_key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception as error:
        raise RuntimeError(f"Groq request failed: {error}") from error
    answer = str(response.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError("Groq returned an empty answer")
    return answer


class HybridRetriever:
    """Dense + BM25 retrieval with page-level evidence aggregation."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        embedding_model: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> None:
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.encoder = SentenceTransformer(embedding_model)
        texts = [self._passage_text(chunk.text) for chunk in chunks]
        vectors = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.bm25 = BM25Okapi([tokenize(text) for text in texts])
        self._rerankers: dict[str, CrossEncoder] = {}

    def _passage_text(self, text: str) -> str:
        if "e5" in self.embedding_model.lower():
            return f"passage: {text}"
        return text

    def _query_text(self, query: str) -> str:
        if "e5" in self.embedding_model.lower():
            return f"query: {query}"
        return query
    def _candidate_indices(
        self,
        query: str,
        *,
        candidate_k: int = 80,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        retrieval_mode: str = "hybrid",
    ) -> list[int]:
        if retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("retrieval_mode must be 'dense', 'bm25', or 'hybrid'")
        candidate_k = min(candidate_k, len(self.chunks))
        lexical = np.asarray(self.bm25.get_scores(tokenize(query)))
        lexical_order = np.argsort(-lexical, kind="stable")[:candidate_k]
        query_vector = self.encoder.encode(
            [self._query_text(query)], normalize_embeddings=True
        ).astype("float32")
        _, dense_order = self.index.search(query_vector, candidate_k)
        fused: dict[int, float] = {}
        if retrieval_mode in {"bm25", "hybrid"}:
            for rank, index in enumerate(lexical_order):
                fused[int(index)] = fused.get(int(index), 0.0) + bm25_weight / (rrf_k + rank + 1)
        if retrieval_mode in {"dense", "hybrid"}:
            for rank, index in enumerate(dense_order[0]):
                fused[int(index)] = fused.get(int(index), 0.0) + dense_weight / (rrf_k + rank + 1)
        return sorted(fused, key=lambda index: (-fused[index], index))

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        candidate_k: int = 80,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        diversify_pages: bool = True,
        diversify_sources: bool = False,
        group_by: str = "page",
        retrieval_mode: str = "hybrid",
        rerank: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> list[dict[str, Any]]:
        candidates = self._candidate_indices(
            query,
            candidate_k=candidate_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            rrf_k=rrf_k,
            retrieval_mode=retrieval_mode,
        )
        if not candidates:
            return []
        candidate_scores = {
            index: float(len(candidates) - rank)
            for rank, index in enumerate(candidates)
        }
        if rerank:
            cross_encoder = self._rerankers.get(rerank_model)
            if cross_encoder is None:
                cross_encoder = CrossEncoder(rerank_model)
                self._rerankers[rerank_model] = cross_encoder
            scores = cross_encoder.predict(
                [(query, self.chunks[index].text) for index in candidates],
                show_progress_bar=False,
            )
            ranked = sorted(
                zip(candidates, scores),
                key=lambda pair: (-float(pair[1]), pair[0]),
            )
            candidates = [index for index, _ in ranked]
            candidate_scores = {
                index: float(score) for index, score in ranked
            }
        if group_by not in {"page", "source", "chunk"}:
            raise ValueError("group_by must be 'page', 'source', or 'chunk'")
        groups: dict[tuple[str, int] | str, list[int]] = {}
        for index in candidates:
            chunk = self.chunks[index]
            key: tuple[str, int] | str = (
                (chunk.source, chunk.page) if group_by == "page"
                else chunk.source if group_by == "source"
                else f"chunk:{index}"
            )
            groups.setdefault(key, []).append(index)
        group_order = sorted(
            groups,
            key=lambda group: min(
                candidates.index(index) for index in groups[group]
            ),
        )
        selected: list[tuple[int, float]] = []
        seen_sources: set[str] = set()
        for group in group_order:
            index = groups[group][0]
            source = self.chunks[index].source
            if (
                diversify_sources
                and source in seen_sources
            ):
                continue
            selected.append((index, candidate_scores[index]))
            seen_sources.add(source)
            if len(selected) == k:
                break
        if not diversify_pages:
            selected = [(index, candidate_scores[index]) for index in candidates[:k]]
        return [
            {**self.chunks[index].as_dict(), "score": score, "rank": rank}
            for rank, (index, score) in enumerate(selected[:k], start=1)
        ]


def load_golden_set(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("Golden set must contain a JSON list")
    return rows


def _legacy_page_targets(
    data_dir: str | Path,
    *,
    chunk_size: int = 1800,
    chunk_overlap: int = 300,
) -> dict[int, tuple[str, int]]:
    """Map old chunk labels to source/page without trusting old chunk positions."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    targets: dict[int, tuple[str, int]] = {}
    legacy_id = 0
    for pdf_path in sorted(Path(data_dir).glob("*.pdf")):
        with fitz.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text()
                if not text.strip():
                    continue
                for _ in splitter.split_text(text):
                    targets[legacy_id] = (pdf_path.name, page_number)
                    legacy_id += 1
    return targets


def evaluate(
    retriever: HybridRetriever,
    golden_rows: Iterable[dict[str, Any]],
    *,
    split: str,
    data_dir: str | Path,
    k: int = 5,
    **search_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate page-level evidence, retaining the old benchmark's split."""
    legacy_targets = _legacy_page_targets(data_dir)
    rows = [
        row
        for row in golden_rows
        if row.get("answerable") and row.get("evaluation_split") == split
    ]
    if not rows:
        raise ValueError(f"No answerable rows found for split {split!r}")
    reciprocal_ranks: list[float] = []
    source_reciprocal_ranks: list[float] = []
    audited_reciprocal_ranks: list[float] = []
    hits = 0
    source_hits = 0
    audited_hits = 0
    audited_rows = 0
    toc_target_rows: list[str | None] = []
    details: list[dict[str, Any]] = []
    toc_pages: set[tuple[str, int]] = set()
    for source in {legacy_targets[chunk_id][0] for row in rows for chunk_id in row.get("relevant_chunk_ids", []) if chunk_id in legacy_targets}:
        with fitz.open(Path(data_dir) / source) as document:
            toc_pages.update(
                (source, page_number)
                for page_number, page in enumerate(document, start=1)
                if is_table_of_contents(page.get_text())
            )
    for row in rows:
        target_pages = {
            legacy_targets[chunk_id]
            for chunk_id in row.get("relevant_chunk_ids", [])
            if chunk_id in legacy_targets
        }
        if row.get("relevant_pages"):
            target_pages = {
                (page["source"], page["page"])
                for page in row["relevant_pages"]
            }
        content_target_pages = {
            target
            for target in target_pages
            if target not in toc_pages
        }
        if content_target_pages:
            audited_rows += 1
        elif target_pages:
            toc_target_rows.append(row.get("id"))
        retrieved = retriever.search(row["question"], k=k, **search_kwargs)
        rank = next(
            (
                index + 1
                for index, chunk in enumerate(retrieved)
                if (chunk["source"], chunk["page"]) in target_pages
            ),
            None,
        )
        target_source = row.get("doc_id")
        source_rank = next(
            (
                index + 1
                for index, chunk in enumerate(retrieved)
                if chunk["source"] == target_source
            ),
            None,
        )
        if rank is not None:
            hits += 1
        if source_rank is not None:
            source_hits += 1
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
        source_reciprocal_ranks.append(
            1.0 / source_rank if source_rank is not None else 0.0
        )
        audited_rank = next(
            (
                index + 1
                for index, chunk in enumerate(retrieved)
                if (chunk["source"], chunk["page"]) in content_target_pages
            ),
            None,
        )
        if audited_rank is not None:
            audited_hits += 1
        if content_target_pages:
            audited_reciprocal_ranks.append(
                1.0 / audited_rank if audited_rank is not None else 0.0
            )
        details.append(
            {
                "id": row.get("id"),
                "question": row["question"],
                "target_pages": sorted(target_pages),
                "content_target_pages": sorted(content_target_pages),
                "retrieved": [
                    {"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]}
                    for c in retrieved
                ],
                "rank": rank,
            }
        )
    return {
        "split": split,
        "count": len(rows),
        "hit_at_k": hits / len(rows),
        "mrr_at_k": float(np.mean(reciprocal_ranks)),
        "source_hit_at_k": source_hits / len(rows),
        "source_mrr_at_k": float(np.mean(source_reciprocal_ranks)),
        "toc_target_rows": toc_target_rows,
        "audited_count": audited_rows,
        "audited_hit_at_k": audited_hits / audited_rows if audited_rows else None,
        "audited_mrr_at_k": (
            float(np.mean(audited_reciprocal_ranks))
            if audited_rows
            else None
        ),
        "details": details,
    }
