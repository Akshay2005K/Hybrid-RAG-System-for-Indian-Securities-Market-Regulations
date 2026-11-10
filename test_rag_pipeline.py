from rag_pipeline import (
    extractive_answer,
    load_sentence_chunks,
    has_strong_match,
    is_table_of_contents,
    normalise_page_text,
    tokenize,
)


def test_tokenizer_preserves_finance_terms():
    terms = tokenize("T+1, UPI, NAV and ASBA")
    assert {"t+1", "upi", "nav", "asba"}.issubset(terms)


def test_page_normalisation_removes_line_break_hyphenation():
    assert normalise_page_text("settle-\nment  funds") == "settlement funds"


def test_toc_detection_requires_toc_shape():
    assert is_table_of_contents("Contents\nIntroduction .... 4\nRules .... 7")
    assert not is_table_of_contents("The contents of the account are protected.")


def test_toc_detection_handles_header_after_pdf_metadata():
    page = "Page 3 of 20\n\n\nTable of Contents\nChapter 1 .... 4\nChapter 2 .... 8"
    assert is_table_of_contents(page)


def test_extractive_answer_returns_text_and_citation():
    results = [
        {
            "text": "When shares are not allotted, the blocked amount is released.",
            "source": "guide.pdf",
            "page": 12,
            "chunk_id": 4,
        }
    ]
    answer = extractive_answer("What happens when shares are not allotted?", results)
    assert answer["grounded"]
    assert "blocked amount is released" in answer["answer"]
    assert answer["citations"] == [
        {"source": "guide.pdf", "page": 12, "chunk_id": 4}
    ]


def test_out_of_scope_question_is_not_a_strong_match():
    results = [{"text": "The investor may apply for shares in an IPO."}]
    assert not has_strong_match("What is Reliance's share price today?", results)
