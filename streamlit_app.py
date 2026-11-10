"""Local Streamlit demo for cited stock-market PDF retrieval."""

from pathlib import Path

import streamlit as st

from src.rag_stock_market import (
    HybridRetriever,
    extractive_answer,
    generate_groq_answer,
    has_strong_match,
    load_sentence_chunks,
    load_manifest,
)


DATA_DIR = Path(__file__).parent / "data" / "raw"
MANIFEST_PATH = Path(__file__).parent / "manifest.csv"


@st.cache_resource(show_spinner="Indexing cleaned PDF pages...")
def get_retriever() -> HybridRetriever:
    chunks = load_sentence_chunks(DATA_DIR)
    return HybridRetriever(chunks)


st.set_page_config(page_title="Stock-market PDF RAG", layout="wide")
st.title("Stock-market PDF RAG")
st.caption("Educational search over public documents. Not investment advice.")

question = st.text_area(
    "Question",
    value="What happens to blocked funds when shares are not allotted in a UPI-based public-issue application?",
    height=90,
)
top_k = st.slider("Number of results (top-k)", min_value=1, max_value=10, value=5)
groq_model = st.text_input("Groq model", value="openai/gpt-oss-20b")

if st.button("Retrieve", type="primary") and question.strip():
    retriever = get_retriever()
    manifest = load_manifest(MANIFEST_PATH)
    results = retriever.search(
        question.strip(),
        k=top_k,
        candidate_k=120,
        bm25_weight=0.3,
        dense_weight=0.7,
        group_by="page",
    )
    if not has_strong_match(question.strip(), results):
        st.warning("No strong match in these documents for this question.")
    else:
        try:
            answer_text = generate_groq_answer(
                question.strip(), results, model=groq_model.strip()
            )
        except (RuntimeError, ValueError) as error:
            st.error(str(error))
            st.stop()
        st.subheader("Answer")
        st.write(answer_text)
        st.caption(
            f"{len(results)} results from "
            f"{len({item['source'] for item in results})} documents"
        )

        grouped: dict[str, list[dict[str, object]]] = {}
        for result in results:
            grouped.setdefault(result["source"], []).append(result)
        for source, source_results in grouped.items():
            metadata = manifest.get(source, {})
            st.markdown(f"**{metadata.get('title') or source}**")
            st.caption(
                f"{metadata.get('publisher', 'Unknown publisher')} · "
                f"{metadata.get('doc_date', 'Date unavailable')} · "
                f"{metadata.get('status', 'status unavailable')}"
            )
            if metadata.get("source_url"):
                st.markdown(f"[Source document]({metadata['source_url']})")
            for result in source_results:
                sentence = extractive_answer(
                    question.strip(), [result], max_sentences=1
                )["answer"]
                if sentence:
                    st.markdown(f"**{sentence}**")
                st.caption(f"Page {result['page']}")
