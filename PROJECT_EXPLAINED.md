# The Whole Project Explained — in Simple Language

**Project:** P3 Corrective RAG — Variant · **Student:** Manish Raje, Roll 67, T.E. COMP D17B
**Course:** Knowledge Graph and Information Retrieval Systems Lab (26CMPE742) · **Partner (Primary):** Roll 66 (Akansha Zambare)

This file explains *everything*: the idea, the words used, every notebook cell (what it does, what it prints, what that means), and every part of the report. Read it top to bottom once and you can explain the project to anyone.

---

## Contents
1. [The idea in plain words](#1-the-idea-in-plain-words)
2. [Words you must know (glossary)](#2-words-you-must-know-glossary)
3. [What exactly was given to me (the task)](#3-what-exactly-was-given-to-me-the-task)
4. [The dataset: CRAG](#4-the-dataset-crag)
5. [The pipeline step by step](#5-the-pipeline-step-by-step)
6. [Every notebook cell explained](#6-every-notebook-cell-explained)
7. [The results and what they mean](#7-the-results-and-what-they-mean)
8. [The report explained section by section](#8-the-report-explained-section-by-section)
9. [Files in this repository](#9-files-in-this-repository)
10. [How to run it](#10-how-to-run-it)
11. [Viva questions and answers](#11-viva-questions-and-answers)

---

## 1. The idea in plain words

**Problem.** A chatbot (LLM) answering factual questions often *makes things up* ("hallucinates"), because it only knows what it memorised during training.

**Usual fix: RAG.** Before answering, we *search* some documents, paste the best pieces into the prompt, and the LLM answers from them. This is **Retrieval-Augmented Generation (RAG)**.

**The weakness of RAG.** If the search returns the *wrong* pieces, the LLM still trusts them and gives a confident wrong answer.

**Corrective RAG (CRAG).** Add a "checker" (the **retrieval evaluator**) between search and answer. It looks at what was retrieved and says:
- ✅ **correct** – the pieces look relevant → use them (after removing weak pieces);
- ❌ **incorrect** – the pieces look useless → throw them away and get knowledge from somewhere else;
- 🤔 **ambiguous** – not sure → use both the pieces and the other source.

In the original paper, "somewhere else" = a **web search**.

**My Variant.** I keep the same checker as my Primary partner, but "somewhere else" = a **Knowledge Graph (KG) lookup in Wikidata** (free, structured facts like *(Christopher Nolan, date of birth, 1970-07-30)*).

**My question.** On 200 questions of the CRAG benchmark: *How often does the checker send questions to the KG (fallback rate), and does accuracy improve — for each type of question?*

**Short answer (my findings).**
- Corrective RAG + KG was the most accurate (18.0 %) but only ~1 point better than standard RAG (17.0 %) and no retrieval (17.3 %).
- The checker sent 17.5 % of questions to the KG.
- For the questions the checker called *incorrect*, the KG really helped (5.9 % → 17.6 % accuracy).
- The big weakness is the checker: it judges whether a page is *about the right topic*, not whether it *contains the answer*, so it rarely asks for help.

---

## 2. Words you must know (glossary)

| Word | Simple meaning |
|---|---|
| **LLM** | Large Language Model — a text-generating AI. Ours: **Qwen2.5-1.5B-Instruct** (1.5 billion parameters, small enough for a 4 GB GPU; the rule was ≤ 3B). |
| **RAG** | Search first, then let the LLM answer using what was found. |
| **Chunk** | A small piece of a document (we use 200 words) — documents are too long to give whole. |
| **Embedding** | A list of numbers that represents the meaning of a text. Similar meaning → similar numbers. |
| **Embedder** | Model that makes embeddings. Ours: **BGE-small-en-v1.5** (33 M parameters). |
| **Dense retrieval** | Find chunks whose embedding is closest (cosine similarity) to the question's embedding. |
| **Cross-encoder / reranker** | Model that reads *question + chunk together* and outputs a relevance score. More accurate than embeddings, slower. Ours: **BGE-reranker-base**, used as the **retrieval evaluator**. |
| **Sigmoid score** | Converts the reranker's raw number into 0–1 (like a probability of relevance). |
| **Thresholds** | Upper = 0.60, lower = 0.15. Best score ≥ 0.60 → correct; < 0.15 → incorrect; between → ambiguous. |
| **Knowledge refinement** | Keep only chunks scoring ≥ 0.15; drop the noise. |
| **Knowledge Graph (KG)** | Facts stored as **triples** *(subject, relation, object)*. |
| **Wikidata** | The free KG behind Wikipedia. Each thing has an ID (e.g. Q42); facts are "claims". |
| **Entity linking** | Matching a name in the question ("Steve Nash") to the right KG item. |
| **Fallback rate** | % of questions sent to the KG = (ambiguous + incorrect) / 200. |
| **Accuracy** | % of answers that contain the gold (correct) answer. |
| **Missing** | % of answers that were "I don't know" (not wrong, just abstaining). |
| **Hallucination** | % of answers that were wrong. |
| **CRAG score** | Accuracy − Hallucination. Punishes wrong answers more than abstaining. Negative = more wrong than right. |
| **EM (Exact Match)** | Answer exactly equals gold after cleanup. Very strict. |
| **F1 (token)** | Word-overlap between answer and gold (partial credit). |
| **Recall@K** | % of questions where *at least one* of the top-K chunks contains the answer. |
| **MRR** | Mean Reciprocal Rank — 1 if the first useful chunk is at rank 1, ½ at rank 2, ⅓ at rank 3 … averaged. |
| **NDCG@10** | Ranking quality in the top 10; rewards useful chunks near the top. 1 = perfect. |
| **Seed** | A number that fixes randomness so results can be repeated. |
| **Mean ± std** | Average over seeds ± how much it varied. |
| **Baseline** | The simple method you compare against (here: *no retrieval*). |
| **Stratified sample** | Sample that keeps the same proportions of each category (question type). |

---

## 3. What exactly was given to me (the task)

From the course sheet:

> **13 · P3 Corrective RAG · Variant (partner Roll 66 (Akansha Zambare)):** *On 200 questions from the CRAG benchmark, build the same evaluator with a KG-lookup fallback. Compare fallback rate and accuracy by question type.*

Common rules I followed:
| Rule | How I followed it |
|---|---|
| Python, free resources, open models; small embedder; LLM ≤ 3B | BGE-small, BGE-reranker-base, Qwen2.5-1.5B; laptop GPU / Colab |
| Use only the stated subset size | Exactly 200 CRAG questions |
| Fix seeds; 3 seeds where results vary, mean ± std | Sample seed 42; generation seeds 0,1,2 for the baseline. *RAG methods: 1 seed (GPU time) — stated honestly as a limitation* |
| Retrieval metrics Recall@5, Recall@10, MRR, NDCG@10; EM/F1 | All reported |
| Always a baseline + cost (time, tokens) | No-retrieval baseline; cost table |
| Report negative results honestly, never invent numbers | Small gain reported as "within noise"; every number comes from the notebook |
| P3 tables/figures: T1, T2, F1, F2 | All present (+ extra T3/F3 by question type for the Variant) |
| Submission: code + README, results CSV, report ≤ 10 pages of content, experiment log | All in this repo |

---

## 4. The dataset: CRAG

**CRAG = Comprehensive RAG Benchmark** by Meta (KDD Cup 2024). Each record has:

| Field | Meaning | Example |
|---|---|---|
| `query` | the question | "who won the oscar best actor in 2019?" |
| `answer` / `alt_ans` | gold answer / other accepted answers | "rami malek" |
| `query_time` | when the question was asked (facts change!) | 03/19/2024 |
| `domain` | finance, sports, music, movie, open | movie |
| `question_type` | one of 8 types (below) | simple |
| `search_results` | 5 real web pages (full HTML) | Wikipedia, IMDb … |

**The 8 question types**

| Type | Meaning | Example idea |
|---|---|---|
| simple | one fact | Who won best actor in 2019? |
| simple_w_condition | one fact with a condition | Stock price of X *on date Y* |
| comparison | compare two things | Which ETF traded more, A or B? |
| aggregation | count/sum several facts | How many Grammys has X won? |
| set | list of things | Members of band X? |
| multi-hop | chain facts | Director of the film starring X in 2010? |
| post-processing | calculate from facts | Average of X's 3 biggest acquisitions? |
| false_premise | the question is based on something untrue → answer must be "invalid question" | Name of X's upcoming fashion line (there is none) |

**My subset.** Two official stratified CRAG subsets (272 questions) → stratified random sample of **200** (seed 42):

| Type | Pool | Sample |
|---|---|---|
| simple | 77 | 56 |
| simple_w_condition | 40 | 29 |
| aggregation | 32 | 24 |
| comparison | 31 | 23 |
| false_premise | 31 | 23 |
| set | 25 | 18 |
| multi-hop | 24 | 18 |
| post-processing | 12 | 9 |
| **Total** | **272** | **200** |

Domains in the sample: finance 49, sports 46, movie 43, open 36, music 26.

---

## 5. The pipeline step by step

```
Question + 5 web pages
   │
   ├─(1) Clean HTML → text → 200-word chunks (40-word overlap)
   ├─(2) BGE-small: embed chunks & question → top-10 chunks
   ├─(3) Evaluator (BGE-reranker): score top-5 → max score s
   │        s ≥ 0.60 → CORRECT     → refined chunks only
   │        s < 0.15 → INCORRECT   → KG facts only
   │        else     → AMBIGUOUS   → refined chunks + KG facts
   ├─(4) KG lookup (only for ambiguous/incorrect):
   │        Qwen lists entity names → Wikidata search → facts → readable triples
   │        → rank triples by similarity to the question → keep 15
   ├─(5) Qwen2.5-1.5B writes a short answer
   └─(6) Score: accurate / missing / hallucinated, EM, F1, by question type
```

**Three methods compared**
1. **No retrieval (baseline):** only the question is given to the LLM.
2. **Standard RAG:** always the top-5 chunks, no checking.
3. **Corrective RAG + KG fallback (mine):** steps 3–4 above.

The prompt tells the LLM: answer in as few words as possible; say **"invalid question"** for a false premise; say **"I don't know"** if unsure. It also gives the `query_time`.

---

## 6. Every notebook cell explained

Notebook: `67_P3_Variant_CorrectiveRAG_KG.ipynb`. Markdown cells are headings; below are the **code cells** in order.

### Cell 0 — Colab setup
- **What it does:** Only on Google Colab: downloads this GitHub repo (code + cached Wikidata answers) and installs packages. On a local PC it does nothing.
- **Output:** install messages or nothing.
- **Significance:** Makes the notebook run on any computer without manual setup.

### Cell 1 — Setup
- **What it does:** Imports libraries; fixes random seeds (42) for Python, NumPy and PyTorch; detects the GPU; creates `data/`, `results/`, `figures/`; names the three models.
- **Output:** `torch 2.11.0+cu128 | device: cuda NVIDIA GeForce RTX 3050 Ti Laptop GPU` (on Colab: *Tesla T4*).
- **Meaning:** A GPU was found, so models run fast. Seeds make the experiment repeatable.

### Cell 2 — Load CRAG and sample 200 questions
- **What it does:** Downloads CRAG subsets 1 and 2 from Hugging Face (the files are several JSON objects in a row, so they are decoded one by one). Groups questions by type, computes how many to take from each type in proportion (quota), randomly samples them (seed 42) and shuffles.
- **Output:** `pool: 272 | sample: 200`, the pool/sample table (Section 4) and domain counts.
- **Meaning / inference:** Every type is represented in the same proportion as the original data, so per-type comparisons are fair. Small types (post-processing: 9) give less reliable per-type numbers.

### Cell 3 — Look at one example
- **What it does:** Prints one question's fields and the titles of its 5 pages.
- **Output:** *"who won the oscar best actor in 2019?"* → *rami malek*, domain movie, type simple; pages like "Academy Award for Best Actor – Wikipedia".
- **Meaning:** Shows the data format: each question comes with its own 5 pages — this is the "search corpus" for that question (CRAG Task 1 setting).

### Cell 4 — Parse HTML and chunk
- **What it does:** For each page: remove `<script>`, `<style>`, menus, headers, footers; keep visible text; squeeze whitespace; take up to 6,000 words; split into 200-word chunks that overlap by 40 words; prefix each chunk with its page title. Then deletes the raw HTML to save memory.
- **Output:** `parsed in 62.4s | chunks total=13036 mean/q=65.2`.
- **Meaning:** 13,036 chunks overall, ~65 per question. Overlap means an answer sitting on a boundary is not cut in half. Page title prefix helps the retriever know what the chunk is about.

### Cell 5 — Embed and retrieve (BGE-small)
- **What it does:** Converts every chunk and the question (with BGE's special query prefix) into normalised embeddings; dot product = cosine similarity; keeps the 10 most similar chunks per question.
- **Output:** `embedding + retrieval: 121.0s`.
- **Meaning:** Fast, cheap dense search. These top-10 chunks are what standard RAG and the evaluator will use.

### Cell 6 — Retrieval quality & metrics
- **What it does:** Marks a chunk **relevant** if it contains the (cleaned) gold answer (or ≥ 75 % of its words for long answers). False-premise questions (gold = "invalid question") are skipped because nothing can "contain" that answer. Computes Recall@5, Recall@10, MRR, NDCG@10.
- **Output:**
  ```
  answerable=177 | answer present anywhere in the 5 pages: 0.565
  Recall@5 0.463   Recall@10 0.514   MRR 0.408   NDCG@10 0.335
  ```
- **Meaning / inference:**
  - Of 177 answerable questions, **only 56.5 % have the answer anywhere in their 5 pages** → for 43.5 % no RAG method can find it in the pages. This is a hard ceiling and one reason accuracy is low.
  - Recall@5 = 0.463 → for 46 % of questions the answer is in the top-5 chunks. Since the maximum possible is 56.5 %, the retriever finds the answer in ~82 % of the cases where it exists.
  - MRR 0.408 → when found, the useful chunk is usually ranked 1st or 2nd.

### Cell 7 — The retrieval evaluator (heart of Corrective RAG)
- **What it does:** For each question, the cross-encoder scores (question, chunk) for the top-5 chunks and turns each into 0–1 with sigmoid. Takes the **maximum** score *s*:
  - *s* ≥ 0.60 → **correct**; *s* < 0.15 → **incorrect**; otherwise → **ambiguous**.
  - Keeps chunks with score ≥ 0.15 (knowledge refinement).
  - Deletes the reranker from GPU memory afterwards (4 GB GPU).
- **Output:**
  ```
  correct 165   ambiguous 18   incorrect 17
  KG fallback rate = 0.175
  ```
- **Meaning:** The checker trusts retrieval for 82.5 % of questions and asks the KG for help on 35 questions (17.5 %). Later (Figure F2) we see it trusts too easily.

### Cell 8 — Load the LLM
- **What it does:** Loads Qwen2.5-1.5B-Instruct in half precision (fp16) on the GPU. Defines `chat()` which formats prompts with the chat template, generates answers in batches of 4, and **counts tokens** (prompt and generated) for the cost table. With a seed it samples (temperature 0.7, top-p 0.9); without a seed it is greedy.
- **Output:** `['Paris']` for "What is the capital of France?".
- **Meaning:** Sanity check — the model works and answers briefly.

### Cell 9 — KG-lookup fallback (my Variant)
- **What it does:**
  1. For the 35 fallback questions, asks Qwen to list the named entities (comma-separated).
  2. For each name (max 3): Wikidata `wbsearchentities` → top matching item (e.g. Q-ID).
  3. `wbgetentities` → that item's claims (facts). Skips external IDs, images, URLs.
  4. Looks up the human-readable labels of properties and values (in batches of 50) → triples like *(Guns N' Roses, country of origin, United States)*.
  5. Ranks all triples by BGE similarity to the question and keeps the top 15.
  6. Every API answer is **cached** in `data/kg_cache.json`; calls are throttled (0.5 s) and HTTP 429 "too many requests" is handled, so reruns are identical and polite to Wikimedia.
- **Output:**
  ```
  KG lookups for 35 questions in 210.5s | entity found for 0.829
  which etf had a higher number of shares traded on the day before yesterday, clsk or bksy?
  ['ETF', 'Clsk', 'Bksy']
  (Commando Luchtstrijdkrachten, short name, CLSK) …
  ```
- **Meaning / inference:** For 82.9 % (29/35) of fallback questions at least one entity was found. The example shows the classic weakness: **"CLSK" is a stock ticker (CleanSpark), but Wikidata's top hit is the Dutch Air Force command abbreviated CLSK** → wrong facts. Entity linking with just the top hit is error-prone, and stock-trading volumes are not in Wikidata anyway.

### Cell 10 — Generate answers for the 3 methods
- **What it does:** Builds prompts:
  - no retrieval → question only;
  - standard RAG → top-5 chunks (each cut to 400 characters to fit the 4 GB GPU);
  - corrective → depends on the label (refined chunks / KG facts / both).

  Generates 200 answers per method: 3 seeds for the baseline, 1 seed (seed 0) for the two RAG methods.
- **Output:** times, e.g. `No retrieval seed 0: 55.7s … Standard RAG seed 0: 588.9s … Corrective seed 0: 579.0s`.
- **Meaning:** RAG prompts are ~5× longer, so ~10× slower. Corrective is slightly cheaper than standard RAG because refinement removes weak chunks.

### Cell 11 — Scoring → Table T1
- **What it does:** For each answer: "I don't know" (or empty) → **missing**; false-premise question → accurate only if it says "invalid question"; otherwise **accurate** if the cleaned gold/alt answer appears in the answer (or word-F1 ≥ 0.5 for multi-word answers); else **hallucinated**. Also EM and F1. Averages per seed, then mean ± std.
- **Output:**

  | Method | Acc % | Missing % | Halluc. % | CRAG score % | EM % | F1 % |
  |---|---|---|---|---|---|---|
  | No retrieval **[BASELINE]** | 17.3 ± 1.4 | 17.2 ± 0.3 | 65.5 ± 1.3 | −48.2 ± 2.8 | 8.8 ± 0.6 | 15.1 ± 0.2 |
  | Standard RAG (top-5) | 17.0 | 11.5 | 71.5 | −54.5 | 7.0 | 14.8 |
  | Corrective RAG + KG | **18.0** | 13.5 | 68.5 | −50.5 | 7.5 | **16.5** |

  ("± nan" in the notebook = only one seed, so no std.)
- **Meaning / inference:**
  - Corrective RAG has the best accuracy and F1, but +1.0 point over standard RAG is **smaller than the baseline's seed variation (±1.4)** → not a clear win.
  - Standard RAG *lowers* "I don't know" (17.2 → 11.5 %) but *raises* hallucination (65.5 → 71.5 %): given pages, the small model becomes over-confident.
  - Corrective RAG reduces hallucination compared to standard RAG (71.5 → 68.5 %).
  - All CRAG scores are negative → a 1.5B model is wrong more often than right on this hard benchmark (published large systems also struggle on CRAG).

### Cell 12 — Table T2 (evaluator labels and fallback)
- **Output:**

  | Label | Count | Share % |
  |---|---|---|
  | correct | 165 | 82.5 |
  | ambiguous | 18 | 9.0 |
  | incorrect | 17 | 8.5 |
  | KG fallback (ambiguous+incorrect) | 35 | 17.5 |
  | KG entity found (of fallbacks) | 29 | 82.9 |
- **Meaning:** The KG is used for about 1 in 6 questions.

### Cell 13 — Variant analysis: by question type (Table T3)
- **What it does:** Per type: number of questions, fallback rate, % labelled incorrect, accuracy of each method. Also accuracy per evaluator label.
- **Output (T3):**

  | Type | n | Fallback % | Incorrect % | No-ret acc % | Std RAG % | Corrective % |
  |---|---|---|---|---|---|---|
  | comparison | 23 | 26.1 | 21.7 | 29.0 ± 9.1 | 21.7 | 26.1 |
  | simple | 56 | 23.2 | 14.3 | 7.7 ± 2.7 | 19.6 | 19.6 |
  | false_premise | 23 | 21.7 | 8.7 | 66.7 ± 5.0 | 21.7 | 26.1 |
  | simple_w_condition | 29 | 20.7 | 0.0 | 6.9 ± 3.4 | 17.2 | 17.2 |
  | aggregation | 24 | 12.5 | 4.2 | 4.2 ± 0.0 | 4.2 | 4.2 |
  | multi-hop | 18 | 11.1 | 5.6 | 9.3 ± 6.4 | 5.6 | 5.6 |
  | set | 18 | 0.0 | 0.0 | 16.7 ± 0.0 | 27.8 | 27.8 |
  | post-processing | 9 | 0.0 | 0.0 | 7.4 ± 6.4 | 11.1 | 11.1 |

  Accuracy by evaluator label (seed 0):

  | Label | Standard RAG | Corrective RAG |
  |---|---|---|
  | correct (165) | 18.2 % | 18.2 % |
  | ambiguous (18) | 16.7 % | 16.7 % |
  | incorrect (17) | **5.9 %** | **17.6 %** |
- **Meaning / inference:**
  - **Comparison** questions are sent to the KG most (26 %) and gain the most (+4.3 points over standard RAG).
  - **False premise:** *no retrieval* is best (66.7 %). Pages about real people make the model accept the false premise (e.g. it "finds" a fashion line). The KG partly helps (21.7 → 26.1 %).
  - **Simple / simple-with-condition / set:** retrieval helps a lot vs. no retrieval (e.g. simple 7.7 → 19.6 %). The evaluator mostly says "correct", so corrective = standard here.
  - **Aggregation / multi-hop / post-processing:** low for all methods — they need combining several facts or arithmetic, which a 1.5B model can't do well; KG triples don't contain pre-computed totals.
  - **Most important:** where the evaluator said *incorrect*, replacing the pages with KG facts **tripled accuracy (5.9 → 17.6 %)** → the fallback *works* when it is triggered; it is just triggered too rarely.
  - Each type has only 9–56 questions, so differences of a few points are not reliable.

### Cell 14 — Figure F1: accuracy by method
- **What it shows:** 3 bars (grey baseline, blue standard RAG, red corrective) with the value on top; error bar on the baseline = std over 3 seeds.
- **Inference:** The bars are almost the same height → retrieval and correction give only small overall gains with this small model.

### Cell 15 — Figure F2: confusion matrix (evaluator vs reality)
- **What it does:** Compares the evaluator label with the **actual retrieval quality**:
  - *good* = answer in the top-5 chunks;
  - *partial* = answer in the pages but not retrieved;
  - *bad* = answer not in the pages;
  - *false premise*.
- **Output:**

  | Evaluator ↓ / Actual → | good | partial | bad | false premise |
  |---|---|---|---|---|
  | correct | 73 | 17 | 57 | 18 |
  | ambiguous | 7 | 1 | 7 | 3 |
  | incorrect | 2 | 0 | 13 | 2 |

  `'correct' label precision = 0.442 | recall = 0.890`
- **Meaning / inference (key insight of the project):**
  - When the evaluator says *correct*, retrieval is really good only **44 %** of the time (73 / 165). In **57** cases it said *correct* although the answer was not in the pages at all.
  - But it catches **89 %** of the truly good retrievals.
  - So the evaluator is **lenient**: the cross-encoder measures *"is this page about the topic?"* (yes — the page is about the right movie/player), not *"does it contain the exact fact?"*. That is why few questions reach the KG. **The bottleneck is the evaluator, not the KG.**
  - When it says *incorrect*, it is usually right (13 of 17 really bad).

### Cell 16 — Figure F3: accuracy and fallback by question type
- **What it shows:** Grouped bars = accuracy of the 3 methods per type; dashed black line (right axis) = fallback rate.
- **Inference:** Fallback is highest for comparison/simple/false-premise and zero for set/post-processing. The biggest bar differences are false premise (no retrieval wins) and simple (retrieval wins).

### Cell 17 — Threshold sensitivity (Table T4)
- **What it does:** Without regenerating answers, recounts labels for other upper/lower thresholds and the precision of "correct".
- **Output (selected rows):**

  | Upper | Lower | correct | ambiguous | incorrect | fallback | precision of "correct" |
  |---|---|---|---|---|---|---|
  | 0.4 | 0.15 | 177 | 6 | 17 | 0.115 | 0.441 |
  | **0.6** | **0.15** | **165** | **18** | **17** | **0.175** | **0.442** |
  | 0.8 | 0.15 | 148 | 35 | 17 | 0.260 | 0.466 |
- **Meaning:** A stricter upper threshold sends more questions to the KG (11.5 % → 26 %) but precision barely improves (0.44 → 0.47) → **changing the threshold cannot fix a checker that measures the wrong thing.** Thresholds were fixed before the experiment (no tuning on test data).

### Cell 18 — Cost (Table T5)
- **Output:**

  | Stage | Time (s) | Prompt tokens | Generated tokens |
  |---|---|---|---|
  | HTML parsing + chunking | 62.4 | – | – |
  | Embedding + retrieval | 121.0 | – | – |
  | Evaluator | 25.9 | – | – |
  | KG lookup (entity extraction + Wikidata) | 210.5 | 2,699 | 209 |
  | Generation: no retrieval (3 seeds) | 171.6 | 74,862 | 4,881 |
  | Generation: standard RAG (1 seed) | 588.9 | 138,362 | 2,202 |
  | Generation: corrective RAG (1 seed) | 579.0 | 121,401 | 2,156 |
- **Meaning:** Whole experiment ≈ 30 min on a laptop GPU, ₹0 cost. Generation dominates. Corrective RAG uses **~12 % fewer prompt tokens** than standard RAG (refinement removes weak chunks). KG time is mostly the deliberate 0.5 s pauses for Wikidata.

### Cell 19 — Save results and print failure cases
- **What it does:** Saves every table to `results/*.csv`, an `all_metrics.csv` with every metric and seed, `per_question_predictions.csv` (every answer), and `evaluator_and_kg_trace.json` (scores and KG facts per question). Prints failure cases.
- **Output examples:**
  - *"what is the name of millie bobby brown's upcoming fashion line?"* gold **invalid question**; evaluator *correct* (0.97); both RAGs answer "Florence by Mills" → pages about her real brand made the model accept the false premise.
  - *"how many grand slams has novak djokovic won on clay courts?"* gold **3**; both answer **18** → the model counted all Grand Slams, not only clay — multi-hop reasoning failure.
- **Significance:** Everything is saved, so every number in the report can be checked.

---

## 7. The results and what they mean

**One-paragraph conclusion.** On 200 CRAG questions, corrective RAG with a Wikidata fallback reached **18.0 %** accuracy vs **17.0 %** for standard RAG and **17.3 ± 1.4 %** for no retrieval — a small, not significant gain. The evaluator sent **17.5 %** of questions to the KG; where it said *incorrect*, the KG **tripled** accuracy (5.9 → 17.6 %). But the evaluator is lenient (precision of "correct" only 44 %), so most bad retrievals never reach the KG. Results vary strongly by question type: comparison questions benefit most, false-premise questions are hurt by any retrieval, and aggregation/multi-hop stay hard for a 1.5B model.

**Why is accuracy low overall?** (good viva answer)
1. Small 1.5B model.
2. Strict automatic scoring (answer must contain the gold string).
3. Only 56.5 % of questions even have the answer in their pages.
4. Many CRAG questions are about fast-changing facts (stock prices, recent games).

---

## 8. The report explained section by section

`67_P3_Variant_Report.pdf` / `.docx` follows the department template.

| Page / Section | What it contains | Why |
|---|---|---|
| Front page | College course front page: roll 67, name, course, faculty, title, DOP/DOS | Required by the lab |
| Blank page | Intentionally empty | As instructed |
| Cover (template) | Project title, SDG 4 & 9 badges, course details, team details, academic year | Template page 1 |
| Table of contents | Auto-generated | Template |
| **Abstract** | 150–250 words: problem, method, models, results, SDG | Quick summary |
| **1 Introduction** | 1.1 why RAG fails and why KGs help; 1.2 CRAG shows RAG struggles, web search is costly; 1.3 users: students, engineers, faculty/partner | Context |
| **2 Problem, objectives, scope** | Exact problem statement; 6 measurable objectives; in/out of scope; assumptions (4 GB GPU, automatic scoring, Wikidata limits) | What exactly is done |
| **3 Literature review** | Table of 10 sources: RAG, Corrective RAG, CRAG benchmark, Self-RAG, BGE, Wikidata, LLM+KG roadmap, GraphRAG, Qwen2.5, RAGAS; 3.1 gap: no one measured how often a CRAG evaluator would use a KG, per question type | Shows background and novelty |
| **4 Design** | 4.1 solution; **Figure 1 architecture diagram**; 4.3 module table (loader, chunker, retriever, evaluator, KG lookup, generator, scorer); 4.4 dataset table (pool/sample); 4.5 pseudocode + design decisions | How it is built |
| **5 Implementation** | 5.1 workflow; 5.2 tools table (versions); 5.3 details of each part and **challenges** (Wikidata rate-limit, 4 GB VRAM); **5.4 screenshots of executed notebook cells with captions**; 5.5 contribution table | How it was done, with evidence |
| **6 Testing & results** | 6.1 test cases TC-01…07 (incl. the one that failed and was fixed); 6.2 metric definitions; **6.3 T1 + F1, T2 + F2, T3 + F3, threshold table, retrieval metrics, cost table, 5 failure cases**; 6.4 how to demo | The main results |
| **7 Sustainability & SDG** | Environmental (one laptop GPU, ~30 min), social (abstaining reduces misinformation; KG bias), economic (₹0); SDG 4 Quality Education, SDG 9 Industry/Innovation; measures; indicators | Mandatory template section |
| **8 Feasibility, limitations, risks** | ₹0 budget table; limitations (200 q, automatic scoring, Wikidata ≠ CRAG KG, top-1 linking, small model, 1 seed for RAG); ethics/privacy (no personal data, CC BY-NC licence, API etiquette) | Honesty |
| **9 Conclusion & future scope** | Summary of findings; future: better entity linking, CRAG mock-KG, tuned thresholds, LLM judge, energy measurement | Wrap-up |
| **References** | 15 IEEE-style references (min. 10 required) | Required |
| **Appendices** | A: more screenshots; B: KG-lookup code; C: folder contents; D: how to run | Extra evidence |
| Final checklist | Ticked submission items | Template |

**The five failure cases in the report (Section 6.3)**
1. *"Who directed Guns N' Roses – Live in Paris?"* (simple, gold: Paul Becher). Evaluator: *ambiguous* (0.20). The KG linked the **band**, not the concert film, so the facts were about the band. Standard RAG honestly said "I don't know", but corrective RAG guessed "Tommy LiPuma" (a producer). *Lesson:* wrong entity linking can make the KG fallback **worse** than abstaining.
2. *"Name of Millie Bobby Brown's upcoming fashion line?"* (false premise, gold: invalid question). Evaluator: *correct* (0.97). Both RAGs answered "Florence by Mills" (her real beauty brand). *Lesson:* on-topic pages hide a false premise.
3. *"How many Grand Slams has Djokovic won on clay?"* (multi-hop, gold: 3). Both answered 18 (all his Slams at the time). *Lesson:* the small model doesn't apply the "clay" condition.
4. *"Average amount Amazon spent on its three largest acquisitions?"* (post-processing, gold: ~$8.68 billion). Evaluator: *correct* (0.98), but the model only explained that it needed more information. *Lesson:* the evaluator is over-confident and the model can't do the arithmetic.
5. *"Who has a longer streak of consecutive No. 1 hits, Ariana Grande or Taylor Swift?"* (comparison, gold: Ariana Grande). Evaluator: *correct* (0.97); both answered "Taylor Swift". *Lesson:* a comparison needs facts about both sides; the pages covered only one well.

**Tables and figures required for P3 and where they are**

| Required | In report | In notebook / repo |
|---|---|---|
| T1 accuracy: no retrieval / standard / corrective | Table 6.2 | Cell 11, `results/T1_accuracy_by_method.csv` |
| T2 evaluator label counts + fallback rate | Table 6.3 | Cell 12, `results/T2_evaluator_labels.csv` |
| F1 bar chart of accuracy by method | Figure F1 | Cell 14, `figures/F1_accuracy_by_method.png` |
| F2 confusion matrix evaluator vs actual quality | Figure F2 | Cell 15, `figures/F2_confusion_matrix.png` |
| *Variant extra:* accuracy & fallback by type | Table 6.4, Figure F3 | Cells 13, 16 |
| Baseline, cost | Tables 6.2, 6.6 | Cells 11, 18 |

---

## 9. Files in this repository

| File / folder | What it is |
|---|---|
| `67_P3_Variant_CorrectiveRAG_KG.ipynb` | The notebook (already executed, outputs saved) |
| `notebook_render.html` | Same notebook as a web page (open in any browser) |
| `67_P3_Variant_Report.pdf` / `.docx` | The report |
| `results/` | `all_metrics.csv` (every metric, every seed), T1–T5 CSVs, per-question answers, evaluator & KG trace |
| `figures/` | F0 architecture, F1, F2, F3, SDG badges |
| `screenshots/` | Images of every executed code cell (used in the report) |
| `data/kg_cache.json` | Saved Wikidata responses (reproducible, offline) |
| `README.md` | Steps and versions |
| `EXPERIMENT_LOG.md` | What was run and when, problems faced, AI-tool use |
| `DEMO_GUIDE.md` | Short script for the live demo |
| `PROJECT_EXPLAINED.md` | This file |
| `notebook_src.py`, `make_nb.py`, `run_nb.py` | Notebook source, builder, cell-by-cell runner |
| `analysis.py`, `build_report.py`, `shots.py`, `topdf.py`, `arch_diagram.py`, `sdg_badges.py` | Scripts that build the report, screenshots and figures |
| `requirements.txt` | Package versions |

---

## 10. How to run it

**Google Colab (easiest, e.g. on a college PC)**
1. Open: `https://colab.research.google.com/github/manishraje28/KGIRS-miniproject/blob/main/67_P3_Variant_CorrectiveRAG_KG.ipynb`
2. Runtime → Change runtime type → **T4 GPU**.
3. Run cells one by one (Shift+Enter) or Runtime → Run all (~15 min). Cell 0 clones the repo and installs everything. Click "Run anyway" if warned.
4. For a quick demo, run cells 0–7 live (~5 min) and scroll through the saved outputs of the rest.

**Local PC with an NVIDIA GPU** — see `README.md`.

*Note:* a fresh run can differ by about a point in the RAG accuracies because answers are sampled; data, retrieval, evaluator and KG steps give identical numbers.

---

## 11. Viva questions and answers

| Question | Answer |
|---|---|
| What is Corrective RAG? | RAG plus an evaluator that labels retrieval correct/ambiguous/incorrect and takes a corrective action (refine, replace, or combine with another source). |
| What is your Variant's change? | The corrective source is a Wikidata knowledge-graph lookup instead of web search. |
| Why Wikidata? | Free, structured, reproducible; the CRAG mock-KG API needs the organisers' server. |
| How does the evaluator decide? | Cross-encoder score (0–1) for each of the top-5 chunks; max ≥ 0.60 correct, < 0.15 incorrect, otherwise ambiguous. |
| What is fallback rate? | Share of questions sent to the KG = (ambiguous + incorrect)/200 = 17.5 %. |
| Did the KG help? | Overall a little (+1 point). Where the evaluator said "incorrect", accuracy went 5.9 % → 17.6 %. |
| Main finding? | The evaluator is the bottleneck — it checks topic, not the fact (precision 44 %), so few questions reach the KG. |
| Which question type benefits most? | Comparison (+4.3 points). False premise is best without retrieval. |
| Why is accuracy low? | 1.5B model, strict scoring, only 56.5 % of questions have the answer in their pages, many fast-changing facts. |
| What are Recall@5 / MRR / NDCG? | Answer in top 5 (46.3 %); 1/rank of first useful chunk averaged (0.408); ranking quality in top 10 (0.335). |
| What is the baseline? | No retrieval — the LLM answers from memory. |
| Why one seed for RAG methods? | GPU time on a throttled 4 GB laptop GPU; stated as a limitation; baseline has 3 seeds (std 1.4). |
| What is the CRAG score and why negative? | Accuracy − hallucination; negative because the small model is wrong more often than right. |
| How did you keep it reproducible? | Fixed seeds, cached Wikidata, saved outputs, pinned package versions. |
| Biggest problems faced? | Wikidata rate-limit (HTTP 429) and GPU memory on a 4 GB card — fixed with throttling/caching and smaller batches. |
| Future work? | Evaluator that checks for the answer (not the topic), better entity linking, CRAG mock KG, tuned thresholds on a dev set, larger model, LLM judge. |
| SDGs? | SDG 4 (cheap assistant that admits "I don't know"), SDG 9 (open AI on free hardware). |
