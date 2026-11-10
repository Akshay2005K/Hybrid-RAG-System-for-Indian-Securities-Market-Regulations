# Architecture

```mermaid
flowchart LR
    PDFs[57 source PDFs] --> Extract[PyMuPDF extraction]
    Extract --> Clean[Normalize text + exclude TOC pages]
    Clean --> Sentence[Sentence-level evidence with page metadata]
    Sentence --> Dense[Normalized dense embeddings]
    Sentence --> BM25[BM25 lexical index]
    Query[User question] --> Fuse[Reciprocal-rank fusion]
    Dense --> Fuse
    BM25 --> Fuse
    Fuse --> Group[Page/source grouping]
    Group --> Results[Retrieved sentences with PDF/page citations]
    Results --> LLM[Local Ollama answer generation]
    LLM --> Demo[Streamlit local demo]
    Results --> Eval[Dev/test evaluation]
```

The notebook is a demonstration client. Production ingestion and retrieval
live in [`src/rag_stock_market/pipeline.py`](./src/rag_stock_market/pipeline.py).
The PDFs are not used to fine-tune the models. Ingestion creates a
sentence-level search index, and retrieval supplies relevant sentences as
context to the pretrained local answer model.
