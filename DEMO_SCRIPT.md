# Demo Script — Colab + Report Side by Side (≈ 15 minutes)

**How to use this file:** follow it top to bottom. Each step tells you
- 🖥️ **COLAB** — what to click or scroll to,
- 📄 **REPORT** — which page / table / figure ma'am should look at (page numbers are from the 24-page PDF),
- 🗣️ **SAY** — the words to say (read them in your own style).

---

## Before ma'am comes (do this 10 minutes early)

1. Open the notebook in Colab (personal Gmail):
   `https://colab.research.google.com/github/manishraje28/KGIRS-miniproject/blob/main/67_P3_Variant_CorrectiveRAG_KG.ipynb`
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run **cell 0** (Colab setup) and **cell 1** (Setup) with **Shift+Enter**. Click "Run anyway" if warned. Wait until cell 1 prints `device: cuda Tesla T4`.
4. Keep the report PDF open in a second tab / window.
5. ⚠️ **Never click "Run all"** and never run the *Generation* cell (cell 10) — it takes 5–10 min. Every cell you *don't* run keeps its old saved output, so you can still show it.

**Colab tips:** each grey box is a cell; the output appears just below it. A cell is running when it shows a spinning circle on its left. Use the **Table of contents** icon (☰ on the left side bar) to jump to a section heading (§1, §2 …).

---

## Part 1 — What the project is (2 minutes, mostly the report)

📄 **REPORT:** page 1 (cover) → page 4 (Abstract).

🗣️ **SAY:**
> "Ma'am, my project is **P3 Corrective RAG – Variant**. My partner Akansha Zambare, Roll 66, did the Primary experiment; I did the Variant.
>
> **The problem:** an LLM answering questions often makes up answers. RAG — retrieval-augmented generation — first *searches* documents and gives the best pieces to the LLM. But if the search brings the wrong pieces, the LLM still trusts them and answers wrongly.
>
> **Corrective RAG** adds a *checker* — called the **retrieval evaluator** — between search and answer. It labels the retrieved pieces as **correct, ambiguous or incorrect**. If they look bad, it takes knowledge from another source.
>
> In the original paper, that other source is **web search**. In **my Variant**, it is a **knowledge-graph lookup in Wikidata**. My task was: on **200 questions from the CRAG benchmark**, use the same evaluator with a KG fallback, and compare the **fallback rate** and **accuracy by question type**."

📄 **REPORT:** page 7, **Figure 1** (architecture diagram). Point at it while saying:

> "This is the full pipeline. Top row: question and its 5 web pages → split into chunks → search with **BGE-small** → the evaluator **BGE-reranker** gives a score.
> Score ≥ 0.60 → **correct** → use the chunks. Score < 0.15 → **incorrect** → throw them away and use **KG facts**. In between → **ambiguous** → use both.
> The purple box is my Variant, the KG lookup. Finally **Qwen2.5-1.5B** writes the answer and I score it.
> I compare three methods: **no retrieval** (baseline), **standard RAG**, and **corrective RAG with KG fallback**."

---

## Part 2 — The notebook, cell by cell (≈ 10 minutes)

### Cell 0 – Colab setup *(already run)*
🖥️ Show cell 0.
🗣️ "This cell downloads my GitHub repo and installs the packages, so the notebook runs on any computer. It also brings a saved copy of the Wikidata answers, so results are reproducible."

### Cell 1 – Setup *(already run)*
🖥️ Show output `device: cuda Tesla T4`.
📄 Report page 8, **Table 5.1** (tools) — or Figure 2 on page 10.
🗣️ "Here I import the libraries, **fix the random seed (42)** so results repeat, and check the GPU. All models are open and small, as the rules say: **BGE-small** embedder, **BGE-reranker-base** as evaluator, and **Qwen2.5 with 1.5 billion parameters** — under the 3-billion limit. Everything runs on free Colab."

### §1 Dataset — Cell 2 *(run it live: Shift+Enter, ~20 s)*
🖥️ Run cell 2. Output: `pool: 272 | sample: 200` and a table per question type.
📄 Report page 8, **Table 4.2** (same table) — and Figure 3 on page 10.
🗣️ "This downloads the **CRAG benchmark** by Meta — real questions, each with 5 real web pages. I took two official subsets, 272 questions, and picked **200 by stratified sampling**: each of the **8 question types** keeps its share — simple, comparison, false premise, aggregation, multi-hop, set, post-processing, simple-with-condition. This is needed because my task is to compare **by question type**. Below are the domains: finance, sports, movie, open, music."

### Cell 3 – Example question *(run live, instant)*
🖥️ Output: *"who won the oscar best actor in 2019?"* → *rami malek*, and 5 page titles.
📄 Appendix Figure A1, page 21.
🗣️ "One example: the question, the correct answer, the date it was asked, its type, and the titles of its 5 web pages. These 5 pages are the only documents we can search for this question."

### Cell 3b – Where the HTML content is *(run live, ~2 s — new cell right after the example)*
🖥️ Output: page *Academy Award for Best Actor – Wikipedia*, `HTML size: 617663 characters`, then three blocks: **RAW HTML**, **CLEAN TEXT**, **FIRST CHUNK**.
📄 Section 5.3 (page 9) describes this step.
🗣️ "This shows what is inside one of the 5 pages. The raw HTML is over 6 lakh characters, mostly tags and styling — useless for the model. After removing scripts, menus and tags, we get clean readable text: about 11,000 words. That is cut into 200-word chunks; this is the first chunk the retriever will search."

### §2 Cell 4 – Parsing and chunking *(run live, ~1 min — talk while it runs)*
🖥️ Output: `chunks total=13036 mean/q=65.2`.
📄 Appendix Figure A2, page 21; method in Section 5.3, page 9.
🗣️ "The pages are raw HTML, so I remove scripts, menus, headers and footers and keep the visible text. Then I cut it into **200-word chunks with 40 words of overlap**, so an answer is not cut in half at a border. In total **13,036 chunks**, about 65 per question."

### Cell 5 – Embedding and retrieval *(run live, ~1 min)*
🖥️ Output: timing.
📄 Appendix Figure A3, page 21.
🗣️ "BGE-small turns every chunk and the question into vectors. Chunks whose vector is closest to the question's vector are the most similar in meaning. I keep the **top 10** for each question."

### §3 Cell 6 – Retrieval metrics *(run live, ~10 s)*
🖥️ Output: `answer present anywhere in the 5 pages: 0.565`, and Recall@5 0.463, Recall@10 0.514, MRR 0.408, NDCG@10 0.335.
📄 Report page 17, **Table 6.6a**; Figure 4 on page 11.
🗣️ "A chunk counts as relevant if it contains the correct answer. False-premise questions are left out because they have no answer.
> First important number: only **56.5 %** of questions even have the answer anywhere in their 5 pages — so no RAG method can answer the rest from the pages.
> **Recall@5 = 0.463**: for 46 % of questions the answer is in the top 5 chunks. **MRR 0.408**: when found, it is usually ranked first or second. **NDCG@10 0.335** measures the ranking quality in the top 10."

### §4 Cell 7 – The evaluator *(run live, ~30 s)* ⭐ core of Corrective RAG
🖥️ Output: `correct 165, ambiguous 18, incorrect 17`, `KG fallback rate = 0.175`.
📄 Report page 15, **Table 6.3 (T2)**; Figure 5 on page 11.
🗣️ "This is the **corrective step**. The cross-encoder reads the question together with each of the top 5 chunks and gives a score between 0 and 1. I take the highest score: **≥ 0.60 correct, < 0.15 incorrect, otherwise ambiguous**. Chunks below 0.15 are dropped — that is called *knowledge refinement*.
> Result: **165 correct, 18 ambiguous, 17 incorrect**. So **35 questions — 17.5 % — go to the knowledge graph**. That is my **fallback rate**, Table T2 in the report."

### §5 Cell 8 – Load Qwen *(run live, ~1–2 min — talk while it downloads)*
🖥️ Output: `['Paris']`.
📄 Appendix Figure A4, page 22.
🗣️ "This loads the LLM, **Qwen2.5-1.5B-Instruct**, on the GPU. I test it with 'What is the capital of France?' and it answers 'Paris'. The function here also **counts every token** — needed for the cost table."

> ✋ **From here on, do NOT run cells. Just scroll and show the saved outputs.** Say: *"The next steps take about 10–15 minutes on the GPU, so I'll show the outputs from my full run."*

### §6 Cell 9 – KG lookup (my Variant) *(show saved output)*
🖥️ Output: `KG lookups for 35 questions … entity found for 0.829`, then the CLSK example with triples.
📄 Report page 12, **Figure 6**; method in Section 5.3, page 9.
🗣️ "This is **my Variant**. For the 35 fallback questions:
> 1. Qwen lists the names in the question,
> 2. I search each name in **Wikidata**,
> 3. I take that item's facts and turn them into **triples** — subject, relation, object,
> 4. I keep the **15 triples most similar** to the question.
> All Wikidata answers are saved in a cache, so the run is reproducible.
> For **82.9 %** of the questions at least one entity was found.
> This example shows a weakness: 'CLSK' is a **stock ticker**, but Wikidata's top match is a **Dutch Air Force command** with the same short name. So wrong facts come in — that is an **entity-linking error**."

### §7 Cell 10 – Generation *(show saved output)*
🖥️ Output: timings per method.
📄 Appendix Figure A5, page 22.
🗣️ "Here all three methods answer all 200 questions. The prompt says: answer in few words, say **'I don't know'** if unsure, and say **'invalid question'** if the question has a false premise. For corrective RAG the context depends on the label — chunks, KG facts, or both. The baseline was run with **3 seeds**; the two RAG methods with **1 seed** because of GPU time — I wrote this in the limitations."

### §8 Cell 11 – Scoring → Table T1 ⭐
🖥️ Output: the T1 table.
📄 Report page 14, **Table 6.2 (T1)** and **Figure F1**; Figure 7 on page 13.
🗣️ "An answer is **accurate** if it contains the correct answer, **missing** if it says 'I don't know', otherwise a **hallucination**. **CRAG score** = accuracy minus hallucination.
> Results: **no retrieval 17.3 ± 1.4 %**, **standard RAG 17.0 %**, **corrective RAG + KG 18.0 %**. Corrective RAG is the best and hallucinates less than standard RAG (68.5 vs 71.5 %). But the gain is only **1 point**, smaller than the seed variation of the baseline, so I report it honestly as **not a clear win**.
> Accuracy is low for all methods because the model is small, the scoring is strict, and only 56 % of questions have the answer in their pages."

### Cell 12 – Table T2
🖥️ Output: label counts + `KG entity found 29 (82.9 %)`.
📄 Page 15, **Table 6.3**.
🗣️ "Same evaluator numbers as a table: 35 fallbacks, Wikidata found an entity for 29 of them."

### §9 Cell 13 – By question type (main Variant result) ⭐
🖥️ Output: the per-type table.
📄 Page 15, **Table 6.4**, and page 16, **Figure F3**.
🗣️ "This is what my task asked — **fallback rate and accuracy by question type**.
> - **Comparison** questions go to the KG most often (**26 %**) and gain the most: **+4.3 points** over standard RAG.
> - **False premise**: *no retrieval* is best, **66.7 %** — when the model sees real pages about a real person, it believes the false premise.
> - **Simple** questions: retrieval helps a lot, **7.7 → 19.6 %**.
> - **Aggregation, multi-hop, post-processing** stay low for every method — a 1.5B model cannot combine many facts.
> - One more important result printed at the bottom: for questions labelled **incorrect**, using KG facts instead of the pages **tripled accuracy, from 5.9 % to 17.6 %**. So the KG fallback **works when it is used**."

### §10 Cells 14–16 – Figures
🖥️ Scroll through the three figures.
📄 Pages 14–16: **F1**, **F2**, **F3**.
- **F1** 🗣️ "Accuracy of the three methods; the bars are almost equal."
- **F2** ⭐ 🗣️ "This is the **key insight**. It compares the evaluator's label with what was *really* in the chunks. When the evaluator says 'correct', the answer is really there only **44 %** of the time — in 57 cases the answer wasn't in the pages at all. The evaluator checks if the page is about the right **topic**, not if it contains the **exact fact**. So it rarely sends questions to the KG — **the weak point is the evaluator, not the KG.**"
- **F3** 🗣️ "Accuracy bars per question type with the fallback rate as the dashed line."

### §11 Cell 17 – Threshold sensitivity
📄 Page 16, **Table 6.5**.
🗣️ "If I make the evaluator stricter — upper threshold 0.8 instead of 0.6 — the fallback rate goes from 17.5 % to 26 %, but its precision hardly improves (0.44 → 0.47). So changing the threshold can't fix a checker that measures topic instead of facts. I fixed the thresholds **before** the experiment, so I didn't tune on test data."

### §12 Cell 18 – Cost
📄 Page 17, **Table 6.6**.
🗣️ "The whole experiment took about **30 minutes** on a laptop GPU and cost **₹0**. Generation takes most of the time. Corrective RAG uses about **12 % fewer prompt tokens** than standard RAG, because refinement removes the weak chunks."

### §13 Cell 19 – Saved results and failure cases
📄 Page 17, **five failure cases** in Section 6.3; GitHub `results/` folder.
🗣️ "Everything is saved as CSV — every metric for every seed — so every number in the report can be checked. Two failure cases printed here:
> - *Millie Bobby Brown's upcoming fashion line* — a false premise, but the model answered with her real brand.
> - *Djokovic's Grand Slams on clay* — correct is 3, the model said 18, it counted all his Slams. That's a multi-hop reasoning failure."

---

## Part 3 — Closing (1 minute, report)

📄 Page 19, **Section 8.2 Limitations** and **9.1 Summary**; page 18 **SDGs**.

🗣️ **SAY:**
> "To conclude: corrective RAG with a Wikidata fallback was **slightly** better — 18 % against 17 % — and it **tripled accuracy on the questions it sent to the KG**. But the evaluator sends too few questions, because it judges topic, not facts. Results depend strongly on question type: comparison benefits most, false-premise is hurt by retrieval.
> **Limitations:** only 200 questions, a small model, strict automatic scoring, Wikidata lacks finance and recent sports facts, and the RAG methods used one seed.
> **Future work:** an evaluator that checks if the answer is really present, better entity linking, and a bigger model.
> It supports **SDG 4** — a free assistant that says 'I don't know' instead of guessing — and **SDG 9** — open AI on free hardware.
> All code, results and the report are on my GitHub."

---

## If something goes wrong

| Problem | What to do / say |
|---|---|
| "Cannot connect to GPU backend" | Use CPU: cells 0–7 still run (a bit slower). Don't run cell 8+; show saved outputs. |
| A cell shows an error | Don't panic: "the saved output from my full run is below / in the report", then scroll on. |
| Numbers slightly different after re-running | "Same data, retrieval and evaluator; the LLM answers are sampled, so they vary by about a point." |
| Internet blocked | Open `notebook_render.html` from GitHub (download and open in a browser) — it has every output. |
| She asks for a cell you didn't run | Its saved output is still visible — just scroll to it. |

## Quick answers if ma'am asks

| Question | Answer |
|---|---|
| What is RAG? | Search documents first, then let the LLM answer using them. |
| What is Corrective RAG? | RAG + an evaluator that checks retrieval and corrects it (refine / replace / combine). |
| What is your change (Variant)? | The corrective source is a Wikidata knowledge graph instead of web search. |
| What is a knowledge graph? | Facts stored as triples (subject, relation, object), e.g. (Christopher Nolan, born, 1970). |
| What is the fallback rate? | % of questions sent to the KG: (ambiguous + incorrect) / 200 = 17.5 %. |
| Why is accuracy so low? | Small 1.5B model, strict scoring, only 56.5 % of questions have the answer in their pages. |
| Baseline? | No retrieval — LLM answers from memory only. |
| Why these models? | Rules: open models, small embedder, LLM ≤ 3B, free resources. |
| Recall@5 / MRR / NDCG? | Answer in top 5 (46.3 %) / average of 1÷rank of first correct chunk (0.408) / ranking quality in top 10 (0.335). |
| Main finding? | The evaluator is the bottleneck — 44 % precision for "correct" — but when the KG is used it triples accuracy. |
| How is it reproducible? | Fixed seeds, cached Wikidata answers, saved outputs, package versions in README. |
| Problems faced? | Wikidata rate limit (HTTP 429) and 4 GB GPU memory — fixed with throttling/caching and smaller batches. |
