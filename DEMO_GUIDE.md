# Demo & Viva Guide — P3 Corrective RAG (Variant), Roll 67

## 1. The project in one minute (say this first)
"RAG systems trust whatever the retriever returns, so when retrieval is bad the LLM hallucinates.
**Corrective RAG** adds an *evaluator* that checks the retrieved passages and labels them **correct / ambiguous / incorrect**.
In the original paper the fix for bad retrieval is a web search. In my **Variant**, the fix is a **knowledge-graph lookup in Wikidata**.
My partner (Roll 12, Primary) runs the reference setup; I test whether the idea holds with a KG fallback on **200 CRAG questions**,
and I compare **fallback rate and accuracy by question type**."

## 2. Pipeline (point at Figure 1 in the report)
| Step | What happens | Tool |
|---|---|---|
| 1 Data | 200 questions sampled from CRAG, stratified by 8 question types (seed 42); each has 5 web pages | CRAG / HF `Quivr/CRAG` |
| 2 Chunk | HTML → text → 200-word chunks (40 overlap); 13,036 chunks | BeautifulSoup |
| 3 Retrieve | Dense search over the question's own pages, top-10 | BGE-small-en-v1.5 |
| 4 Evaluate | Cross-encoder scores top-5; max ≥ 0.60 → correct, < 0.15 → incorrect, else ambiguous | BGE-reranker-base |
| 5 Correct | correct → refined chunks; incorrect → KG only; ambiguous → chunks + KG | — |
| 6 KG lookup | LLM extracts entities → Wikidata search → claims → triples → top-15 by similarity | Wikidata API |
| 7 Generate | Short answer, "I don't know" if unsure, "invalid question" for false premise | Qwen2.5-1.5B-Instruct |
| 8 Score | accurate / missing / hallucinated, EM, F1, per question type | pandas |

Baselines: **no retrieval** (question only) and **standard RAG** (always top-5, no evaluator).

## 3. Live demo — what to show, in order (~8 min)
There is no web frontend: this is a research experiment, so the "frontend" is the **notebook** and its outputs.

1. **GitHub repo** → show folder structure and README (code, results CSV, report, log = all submission items).
2. **Open the notebook** (`67_P3_Variant_CorrectiveRAG_KG.ipynb`) in Colab or VS Code. It is already executed, so every output is visible — **do not re-run during the demo** (full run ≈ 30 min on a laptop GPU).
   Scroll through these cells:
   - Setup cell → GPU detected, seeds fixed.
   - Data cell → table of 272 pool / 200 sample per question type.
   - Example record → a real CRAG question and its 5 pages.
   - Retrieval metrics → Recall@5 / Recall@10 / MRR / NDCG@10.
   - Evaluator cell → **165 correct, 18 ambiguous, 17 incorrect → fallback rate 17.5 %**.
   - KG cell → an example question with the Wikidata triples it retrieved (show it linked "CLSK" to a Dutch air-force command — a real entity-linking failure, good talking point).
   - T1 cell → accuracy table. Figures F1, F2, F3 render inline.
   - Failure-case cell → printed examples.
3. **Optional quick re-run to prove it is live:** run only the first data/sampling cells (≈ 1 min) and show the same 200-question table appears (same seed → same sample).
4. **Open the report PDF** and walk through Section 6 (results) using the talking points below.

## 4. Results — what each table/figure means
**T1 (accuracy):** no retrieval 17.3 ± 1.4 %, standard RAG 17.0 %, corrective RAG + KG 18.0 %.
→ Corrective is the best, but by only ~1 point — within noise. Honest negative result.
Hallucination: 65.5 / 71.5 / 68.5 %. Corrective RAG hallucinates less than standard RAG.

**T2 (evaluator labels):** 165 / 18 / 17 → KG used for 35 questions (17.5 %); Wikidata found an entity for 82.9 % of them.

**F2 (confusion matrix):** when the evaluator says "correct", the answer is actually in the top-5 chunks only ~44 % of the time.
→ **Key insight:** the cross-encoder checks *topic* relevance (page is about the right entity), not whether the *specific fact* is present. So the evaluator is too lenient and sends too few questions to the KG. The bottleneck is the evaluator, not the fallback.

**T3 / F3 (by question type — the Variant's main analysis):**
- Comparison: highest fallback (26.1 %) and corrective RAG helped most (+4.3 pts over standard RAG).
- False premise: *no retrieval is best* (66.7 %) — retrieved pages about real entities make the model accept a false premise.
- Simple / set / with-condition: retrieval helps a lot vs. no retrieval (e.g. simple 7.7 → 19.6 %).
- Aggregation, multi-hop, post-processing: low for every method — a 1.5 B model can't combine many facts.

**Threshold table:** raising the upper threshold sends more questions to the KG; thresholds were fixed beforehand, not tuned on test data.

**Cost (T5):** total ≈ 30 min on RTX 3050; generation dominates; corrective RAG uses ~12 % fewer prompt tokens than standard RAG because refinement drops low-score chunks.

## 5. Likely viva questions
- *Why Wikidata and not CRAG's mock KG?* The mock-KG API needs the organisers' server; Wikidata is free, open and reproducible (responses cached).
- *Why accuracy so low?* 1.5 B model, strict string-match scoring, and only 56.5 % of answerable questions even have the answer in their 5 pages.
- *Why 1 seed for RAG?* GPU time on a throttled 4 GB laptop GPU; stated as a limitation. Baseline has 3 seeds (std 1.4).
- *What would you improve?* Better evaluator (check answer presence, LLM judge), better entity linking, tune thresholds on a dev split, use CRAG mock KG for finance/sports.
- *SDG?* SDG 4 (low-cost assistant that says "I don't know" instead of hallucinating), SDG 9 (open, small-model AI on free hardware).
