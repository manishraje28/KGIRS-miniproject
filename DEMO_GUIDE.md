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

## 6. Cell-by-cell script (what to click, what appears, what to say)

| # | Cell (heading in notebook) | What the code does | Output you will see | What to say |
|---|---|---|---|---|
| 0 | Colab setup | Clones this repo, installs packages | (nothing / install logs) | "Makes the notebook run anywhere; it also brings the cached Wikidata responses." |
| 1 | Setup | Imports, fixes seeds (42), detects GPU, names the 3 models | `device: cuda …` | "Free GPU, open models only: BGE-small embedder, BGE-reranker evaluator, Qwen2.5-1.5B (≤3B) generator." |
| 2 | §1 Dataset | Downloads 2 CRAG subsets (272 q), stratified sample of 200 by question type | pool vs sample table; domain counts | "Stratified so all 8 question types are represented — needed for the by-type analysis." |
| 3 | Example record | Prints one question and its 5 page titles | Oscar Best Actor 2019 → rami malek | "Each question comes with 5 real web pages — that is the retrieval corpus (CRAG Task 1)." |
| 4 | §2 Parsing | HTML → clean text → 200-word chunks, 40 overlap | `chunks total=13036 mean/q=65.2` | "Overlap prevents an answer being split across chunk borders." |
| 5 | Retrieval | BGE-small embeds chunks and query, cosine similarity, keep top-10 | timing | "Dense retrieval per question, over its own pages only." |
| 6 | §3 Retrieval metrics | Chunk is relevant if it contains the gold answer; Recall@5/10, MRR, NDCG@10 | R@5 0.463, R@10 0.514, MRR 0.408, NDCG 0.335; answer in pages 0.565 | "43 % of answerable questions don't even have the answer in their pages — a hard ceiling for any RAG." |
| 7 | §4 Evaluator | Cross-encoder scores top-5 chunks; max ≥0.60 correct, <0.15 incorrect, else ambiguous; drop chunks <0.15 (refinement) | 165 / 18 / 17, fallback 0.175 | "This is the corrective step. Same evaluator as my Primary partner; only 17.5 % go to the KG." |
| 8 | §5 Generator | Loads Qwen 1.5B, defines batched generation + token counting | `['Paris']` | "Sanity check; I also count every prompt/generated token for the cost table." |
| 9 | §6 KG lookup | LLM extracts entity names → Wikidata search → claims → readable triples → top-15 by similarity; cached | entity found 82.9 %; CLSK example | "My Variant. Free, structured, reproducible. The CLSK example is an entity-linking error — a stock ticker linked to a Dutch air-force command." |
| 10 | §7 Generation | Builds prompts for 3 methods and generates answers | timings per method/seed | "Corrective prompt depends on the label: chunks / KG facts / both. Prompt says answer briefly, 'I don't know' if unsure, 'invalid question' for false premise." |
| 11 | §8 Scoring → T1 | accurate / missing / hallucinated, CRAG score, EM, F1 | 17.3 / 17.0 / 18.0 % | "Corrective is best but by ~1 point — within noise. It hallucinates less than standard RAG. Honest negative result." |
| 12 | T2 | Label counts + fallback rate + entity-found rate | table | "35 questions used the KG; Wikidata found an entity for 29." |
| 13 | §9 By question type | Fallback rate and accuracy per type; accuracy per evaluator branch | T3 table | "Comparison: most fallback and biggest gain. False premise: no retrieval best — retrieved pages make the model accept the false premise." |
| 14 | F1 | Bar chart accuracy by method | figure | "Baseline in grey; error bar = std over 3 seeds." |
| 15 | F2 | Confusion matrix: evaluator label vs real retrieval quality | heatmap; precision 0.442, recall 0.890 | "Key insight: evaluator judges topic, not fact presence — 'correct' is right only 44 % of the time, so too few questions reach the KG." |
| 16 | F3 | Accuracy bars + fallback-rate line per type | figure | "The Variant's main figure." |
| 17 | §11 Thresholds | Label counts for other upper/lower thresholds | table | "Stricter threshold → more KG fallback. Thresholds fixed in advance, not tuned on test data." |
| 18 | §12 Cost | Time and tokens per stage | table | "≈30 min total on a laptop GPU; generation dominates; corrective uses ~12 % fewer prompt tokens than standard RAG." |
| 19 | §13 Save + failures | Writes all CSVs/JSON, prints failure cases | Millie Bobby Brown false premise; Djokovic clay slams (gold 3, answered 18) | "Failures: false premise accepted, and multi-hop reasoning — the model counted all slams, not clay." |
