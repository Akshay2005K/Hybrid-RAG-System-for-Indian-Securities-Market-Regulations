"""Reusable stock-market PDF retrieval package."""

from .pipeline import (
    Chunk,
    HybridRetriever,
    evaluate,
    extractive_answer,
    generate_groq_answer,
    generate_ollama_answer,
    has_strong_match,
    is_table_of_contents,
    load_chunks,
    load_golden_set,
    load_manifest,
    load_sentence_chunks,
    normalise_page_text,
    tokenize,
)

__all__ = [
    "Chunk",
    "HybridRetriever",
    "evaluate",
    "extractive_answer",
    "generate_groq_answer",
    "generate_ollama_answer",
    "has_strong_match",
    "is_table_of_contents",
    "load_chunks",
    "load_golden_set",
    "load_manifest",
    "load_sentence_chunks",
    "normalise_page_text",
    "tokenize",
]
