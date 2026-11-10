# Architecture

```mermaid
flowchart LR
    PDFs[57 Source PDFs] --> Extract[PyMuPDF Extraction]
    Extract --> Clean[Text Normalization and TOC Filtering]
    Clean --> Sentence[Sentence-Level Evidence with Page Metadata]

    Sentence --> Dense[Dense Embeddings and FAISS Index]
    Sentence --> BM25[BM25 Lexical Index]

    Query[User Question] --> DenseSearch[Dense Retrieval]
    Query --> BM25Search[BM25 Retrieval]

    Dense --> DenseSearch
    BM25 --> BM25Search

    DenseSearch --> Fuse[Reciprocal Rank Fusion]
    BM25Search --> Fuse

    Fuse --> Group[Page and Source-Level Grouping]
    Group --> Results[Retrieved Evidence with PDF/Page Citations]

    Results --> LLM[Groq-Powered LLM Answer Generation]
    LLM --> Demo[Streamlit Application]

    Results --> Eval[Development and Test Evaluation]
```

The notebook serves as a demonstration and experimentation client. The core ingestion and retrieval logic is implemented in [`src/rag_stock_market/pipeline.py`](./src/rag_stock_market/pipeline.py).

The system processes 57 source PDFs and builds a sentence-level evidence index containing document and page-level metadata. It uses dense vector retrieval with FAISS alongside BM25 lexical retrieval, combining their results through Reciprocal Rank Fusion (RRF). Retrieved evidence is then grouped by source and page to preserve document-level traceability.

The retrieved sentences are supplied as contextual evidence to a Groq-powered LLM, which generates responses grounded in the retrieved regulatory documents and their source references.

The PDFs are not used to fine-tune or train the underlying language or embedding models. Instead, they serve as the knowledge base for retrieval-augmented generation (RAG). The system's development and test evaluation workflows are maintained separately to assess retrieval performance.
