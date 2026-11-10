# Stock-market PDF RAG

The production retrieval code is in
[`src/rag_stock_market/pipeline.py`](./src/rag_stock_market/pipeline.py).
It is independent of notebook execution order and keeps the final test split
out of tuning.

## Local demo

Run the local Streamlit question-answering demo:

```powershell
streamlit run streamlit_app.py
```

It indexes the local PDFs as sentence-level evidence, retrieves the closest
sentences, and uses Groq to answer from those sentences. The
display groups results by document with publisher, date, status, and
source-link metadata. The answer model is not fine-tuned on the PDFs; the PDFs
are retrieved context supplied to the pretrained model.
See [`ARCHITECTURE.md`](./ARCHITECTURE.md) and [`RESULTS.md`](./RESULTS.md)
for the system diagram and benchmark summary.

### Groq setup

Set `GROQ_API_KEY` in your environment, then start the app:

```powershell
streamlit run streamlit_app.py
```

The model name can be changed in the app. If the key or Groq package is
unavailable, the app shows an explicit setup error.

## Run the evaluation

From the project directory:

```powershell
python evaluate_rag.py --data-dir data/raw --gold Golden_set_300.json
```

The script selects the BM25/dense fusion weight, candidate depth, and
page/source grouping using only the 158 development
questions, then evaluates the selected configuration once on the 42-question
test split. It reports both page-level and document-level metrics. Page
grouping is preferred when it improves exact evidence ranking; document
level remains useful for document identification because some legacy
chunk IDs resolve to table-of-contents pages rather than answer evidence.
Results and per-question failures are saved to
`evaluation_results.json`.
The output also includes `audited_hit_at_k` and `audited_mrr_at_k`, which
exclude legacy labels that point to pages automatically identified as TOC
pages. Original `hit_at_k` and `mrr_at_k` remain unchanged for comparability.

The system intentionally does not use a cross-encoder reranker. It aggregates
candidate evidence by source page first, which matches the benchmark's
document/page relevance labels and avoids expensive, opaque reranking.

## Leakage safeguards

- The test split is never used to select a configuration.
- Evaluation labels are converted from legacy chunk IDs to source/page
  evidence, so changing chunk size does not silently invalidate evaluation.
- Sentence records retain source, page, and section metadata for citations.
- Dense vectors are normalized and searched with inner product (cosine
  similarity).
- Lexical retrieval preserves finance terms such as `T+1`, `UPI`, `NAV`, and
  `ASBA`.
- Results are diversified across pages to avoid returning five overlapping
  copies of one passage.

The notebook remains a demonstration client. New experiments should use the
package and commands above rather than relying on notebook cell state.

## Benchmark definition and ablations

`Golden_set_300.json` contains 200 answerable labeled rows: 158 development
and 42 test questions. Each row stores a legacy `relevant_chunk_ids` label
created with the original 1,800-character chunk size and 300-character
overlap. The evaluator maps those IDs back to source PDF and page before
scoring, because the production index uses different cleaned chunks.

Run the clean, same-split ablation benchmark with:

```powershell
python run_ablations.py
```

This writes `Golden_set_300_clean.json` and `ablation_results.json`. The clean
test set has 39 questions: source-selection, conflict-resolution, and
adversarial meta-questions are excluded; multi-document questions remain.
These generated outputs are intentionally ignored by Git; regenerate them
when reproducing the benchmark.
The benchmark compares dense-only, BM25-only, hybrid, and hybrid plus the
cross-encoder reranker on the same questions. Hit@5 and MRR@5 include
question-level percentile bootstrap 95% confidence intervals.

## Failure analysis and model experiments

`failure_audit.json` classifies the 20 clean-test misses from the selected
MiniLM configuration: 10 retrieve the correct PDF but not the labeled page,
and 10 retrieve a different PDF. Several broad document-summary questions
have a single-page gold label even though their wording asks about an entire
document; these are label-quality issues rather than purely retrieval errors.

`fix_gold_mapping.py` creates `Golden_set_300_mapped.json` from the original
gold set. It corrects only six verified TOC/front-matter/introduction-page
mappings; the original file is preserved.

`audit_and_repair_gold.py` performs the full-row rule audit and writes
`Golden_set_300_rule_repaired.json`. It requires every target to belong to the
question's `doc_id` or supporting documents and excludes TOC pages. The
five TOC-only mismatches are repaired by lexical evidence over non-TOC pages.

The original clean-test result was 48.72% page Hit@5. After failure review,
the manually corrected mapping produced 53.85%; the fully rule-repaired
mapping produced 51.28% (39 questions). These corrected results are
post-review and therefore are not untouched holdout measurements. Bootstrap
95% intervals for the rule-repaired result are approximately 35.90--66.67%
for Hit@5 and 25.21--53.51% for MRR@5.

The current environment reports `torch.cuda.is_available() == False`
(`torch 2.10.0+cpu`), so the embedding and reranker experiments ran on CPU.

`model_experiment_results.json` records development-only embedding/chunking
experiments. MiniLM with 1,400-character chunks was stronger than the tested
BGE-base 1,000-character plus section-heading variant. E5-base was installed
and attempted, but CPU throughput was too slow for a complete development
run. `reranker_diagnostic_results.json` records a bounded BGE-reranker test;
it underperformed hybrid retrieval and was not selected.

The current labeled corpus contains 200 answerable rows, of which 188 remain
after removing the three meta-question categories. The clean final test has
39 questions. No new independent labeled split exists in the supplied data;
the original 42-question test split remains untouched for final reporting.
