# Results

The figures below come from the saved benchmark artifacts and are retrieval
metrics at `k=5`.

| Evaluation | Questions | Hit@5 | MRR@5 | Notes |
|---|---:|---:|---:|---|
| Original clean test labels | 39 | 48.72% | 35.85% | Original labels |
| Failure-reviewed labels | 39 | 53.85% | 40.47% | Post-failure label review |
| Rule-repaired labels | 39 | 51.28% | 39.19% | Post-review, rule-repaired labels |
| Dense-only ablation | 39 | 38.46% | 24.23% | Same clean split |
| BM25-only ablation | 39 | 41.03% | 31.07% | Same clean split |
| Hybrid ablation | 39 | **48.72%** | **36.62%** | Same clean split |
| Hybrid + reranker | 39 | 41.03% | 28.08% | MS MARCO reranker |

The corrected-label figures are not untouched holdout measurements because
the labels were reviewed after inspecting failures. A new blind test requires
new questions labeled before retrieval is run.
