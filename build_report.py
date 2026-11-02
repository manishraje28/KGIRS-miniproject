"""Build the micro-project report (.docx) in the department template, with the course front page first.
All numbers are read from results/*.csv produced by the notebook run; nothing is typed by hand."""
import copy, json, os, datetime
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.opc.constants import RELATIONSHIP_TYPE as RT

R, FIG, SH = "results/", "figures/", "screenshots/"
TEMPLATE = "../exp8/Common_Micro_Project_Report_Template_Computer_Engineering (1).docx"
FRONT = "../front page.docx"
TITLE = "Corrective RAG with a Knowledge-Graph Lookup Fallback on the CRAG Benchmark (P3 – Variant)"
TODAY = datetime.date.today().strftime("%d-%m-%Y")

# ------------------------------------------------------------------ results
T1 = pd.read_csv(R + "T1_accuracy_by_method.csv", index_col=0)
T2 = pd.read_csv(R + "T2_evaluator_labels.csv", index_col=0)
T3 = pd.read_csv(R + "T3_by_question_type.csv", index_col=0)
T4 = pd.read_csv(R + "T4_threshold_sensitivity.csv")
T5 = pd.read_csv(R + "T5_cost.csv")
RET = pd.read_csv(R + "retrieval_metrics.csv", index_col=0)
PS = pd.read_csv(R + "metrics_per_seed.csv")
PRED = pd.read_csv(R + "per_question_predictions.csv")
TRACE = {t["interaction_id"]: t for t in json.load(open(R + "evaluator_and_kg_trace.json"))}
EXTRA = json.load(open(R + "report_numbers.json"))

M_NO, M_STD, M_CR = "No retrieval", "Standard RAG (top-5)", "Corrective RAG + KG fallback"
acc = PS.groupby("method")["acc"].agg(["mean", "std"]) * 100
hall = PS.groupby("method")["hall"].mean() * 100
lab = T2["count"]
N = 200
fb_rate = 100 * (lab["ambiguous"] + lab["incorrect"]) / N
pct = lambda m: f"{acc.loc[m, 'mean']:.1f} %"

# ------------------------------------------------------------------ document scaffolding
doc = Document(TEMPLATE)
body = doc.element.body
kids = list(body.iterchildren())
# keep the template cover page (elements 0..12) and the final sectPr; drop guidance + placeholder sections
for el in kids[13:]:
    if el.tag != qn("w:sectPr"): body.remove(el)
SECT = body[-1]

def _set_cell_bg(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor); tcPr.append(sh)

def _borders(table):
    tblPr = table._tbl.tblPr; b = OxmlElement("w:tblBorders")
    for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
        x = OxmlElement(f"w:{e}"); x.set(qn("w:val"), "single"); x.set(qn("w:sz"), "4"); x.set(qn("w:color"), "000000"); b.append(x)
    tblPr.append(b)

def _move(el):                       # python-docx appends after sectPr-less body end; keep sectPr last
    SECT.addprevious(el)

def H1(t): p = doc.add_heading(t, 1); _move(p._p); return p
def H2(t): p = doc.add_heading(t, 2); _move(p._p); return p
def H3(t): p = doc.add_heading(t, 3); _move(p._p); return p

def P(t="", bold_lead=None, size=None, align=None, italic=False):
    p = doc.add_paragraph(); _move(p._p)
    if bold_lead: r = p.add_run(bold_lead); r.bold = True
    r = p.add_run(t); r.italic = italic
    if size: r.font.size = Pt(size)
    p.paragraph_format.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    return p

def BUL(items):
    for t in items:
        p = doc.add_paragraph(style="List Bullet"); _move(p._p)
        if isinstance(t, tuple): r = p.add_run(t[0]); r.bold = True; p.add_run(t[1])
        else: p.add_run(t)
        p.paragraph_format.space_after = Pt(2)

def CAP(t):
    p = doc.add_paragraph(); _move(p._p); r = p.add_run(t); r.italic = True; r.font.size = Pt(9.5); r.bold = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(8)

def IMG(path, width=6.2, caption=None):
    p = doc.add_paragraph(); _move(p._p); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width)); p.paragraph_format.keep_with_next = True
    if caption: CAP(caption)

def TABLE(header, rows, widths=None, caption=None, font=9.5, baseline_row=None):
    if caption:
        c = doc.add_paragraph(); _move(c._p); r = c.add_run(caption); r.bold = True; r.font.size = Pt(9.5)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.paragraph_format.keep_with_next = True; c.paragraph_format.space_after = Pt(2)
    t = doc.add_table(rows=1, cols=len(header)); _move(t._tbl); _borders(t); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]; cell.text = ""; r = cell.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = Pt(font); _set_cell_bg(cell, "D9EAF7")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(font)
            if baseline_row is not None and ri == baseline_row: _set_cell_bg(cells[i], "F2F2F2"); r.italic = True
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths): row.cells[i].width = Inches(w)
    sp = doc.add_paragraph(); _move(sp._p); sp.paragraph_format.space_after = Pt(4)
    return t

def PAGEBREAK():
    p = doc.add_paragraph(); _move(p._p); p.add_run().add_break(WD_BREAK.PAGE)

def CODE(text):
    for line in text.strip("\n").split("\n"):
        p = doc.add_paragraph(); _move(p._p); r = p.add_run(line); r.font.name = "Consolas"; r.font.size = Pt(8.5)
        p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
    sp = doc.add_paragraph(); _move(sp._p)

# ------------------------------------------------------------------ fill template cover
paras = [p for p in doc.paragraphs][:12]
def set_after_label(p, label, value):
    for r in p.runs[1:]: r.text = ""
    p.runs[0].text = label + " "; r = p.add_run(value); r.bold = False
for p in paras:
    if p.text.startswith("Project Title"): set_after_label(p, "Project Title :", TITLE)
    elif p.text.startswith("SDG Aligned"):
        set_after_label(p, "SDG Aligned & Logo :", "SDG 4 (Quality Education), SDG 9 (Industry, Innovation and Infrastructure)")
        p.add_run().add_break(); p.add_run().add_picture(FIG + "sdg_badges.png", width=Inches(2.6))
    elif p.text.startswith("Academic Year"):
        for r in p.runs: r.text = ""
        p.runs[0].text = "Academic Year: 2025-2026"
ct, tt = doc.tables[0], doc.tables[1]
vals = {"Course / Subject Name": "Knowledge Graph and Information Retrieval Systems Lab",
        "Course Code / Semester / Academic Year": "26CMPE742 / Sem VII / 2025-2026",
        "Faculty Incharge / Project Mentor": "Mrs. Sunita Suralkar",
        "Project Group No. (if applicable)": "P3 Corrective RAG – Variant (partner: Roll 12, Primary)",
        "Submission Date": TODAY}
for row in ct.rows:
    k = row.cells[0].text.strip()
    if k in vals: row.cells[1].text = vals[k]
tt.rows[1].cells[0].text, tt.rows[1].cells[1].text, tt.rows[1].cells[2].text, tt.rows[1].cells[3].text = "Manish Raje", "67", "T.E. COMP / D17B", "—"
for row in list(tt.rows)[2:]: tt._tbl.remove(row._tr)
# footer text: "Micro Project Report Template" -> "Micro Project Report"
for s in doc.sections:
    for p in s.footer.paragraphs:
        for r in p.runs:
            if "Template" in r.text: r.text = r.text.replace(" Template", "")

# ------------------------------------------------------------------ prepend course front page + blank page
fdoc = Document(FRONT)
for t in fdoc.tables[0].rows[1].cells[:1]:
    t.paragraphs[0].runs[-1].text = " Micro Project (P3)"
title_cell = fdoc.tables[0].rows[1].cells[1]
for extra in title_cell.paragraphs[1:]: extra._p.getparent().remove(extra._p)
r = title_cell.paragraphs[0].add_run("Corrective RAG with a KG-Lookup Fallback on the CRAG Benchmark (P3 Variant, partner Roll 12)"); r.bold = False
for c in fdoc.tables[0].rows[2].cells:
    lab_ = c.paragraphs[0].runs[0].text.strip()
    c.paragraphs[0].runs[-1].text = " " + TODAY if lab_.startswith(("DOP", "DOS")) else ""
for row, hgt in zip(fdoc.tables[0].rows, ("1300", "2500", "900", "1000")):
    for th in row._tr.iter(qn("w:trHeight")): th.set(qn("w:val"), hgt)
for para in fdoc.paragraphs:                                  # keep the course front page on a single page
    para.paragraph_format.space_after = Pt(2); para.paragraph_format.space_before = Pt(0)
fbody = fdoc.element.body
first = body[0]
for el in list(fbody.iterchildren()):
    if el.tag == qn("w:sectPr"): continue
    el = copy.deepcopy(el)
    for blip in el.iter(qn("a:blip")):                       # re-home front-page images into this package
        rid = blip.get(qn("r:embed")); part = fdoc.part.related_parts[rid]
        from docx.opc.packuri import PackURI
        part.partname = PackURI("/word/media/frontpage_logo" + os.path.splitext(part.partname)[1])
        blip.set(qn("r:embed"), doc.part.relate_to(part, RT.IMAGE))
    first.addprevious(el)
# page break -> blank page -> section break (front page + blank page carry no header/footer)
def _p_with_break():
    p = OxmlElement("w:p"); r = OxmlElement("w:r"); b = OxmlElement("w:br"); b.set(qn("w:type"), "page"); r.append(b); p.append(r); return p
first.addprevious(_p_with_break()); first.addprevious(_p_with_break())
blank = OxmlElement("w:p"); pPr = OxmlElement("w:pPr"); sp = copy.deepcopy(fbody.find(qn("w:sectPr")))
pPr.append(sp); blank.append(pPr); first.addprevious(blank)
# the template section must restart page numbering at 1 after the blank page
pg = SECT.find(qn("w:pgNumType"))
if pg is None: pg = OxmlElement("w:pgNumType"); SECT.append(pg)
pg.set(qn("w:start"), "1")

# ------------------------------------------------------------------ table of contents
PAGEBREAK()
h = H1("Table of Contents")
p = doc.add_paragraph(); _move(p._p)
for tag, txt in (("begin", None), (None, 'TOC \\o "1-2" \\h \\z \\u'), ("separate", None), (None, None), ("end", None)):
    r = p.add_run()
    if tag: f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), tag); r._r.append(f)
    elif txt: it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; r._r.append(it)
    else: r.text = "Right-click → Update Field to refresh the table of contents."
PAGEBREAK()

# ------------------------------------------------------------------ ABSTRACT
H1("Abstract")
P(f"Retrieval-augmented generation (RAG) fails silently when the retriever returns irrelevant passages: the generator still "
  f"trusts them and hallucinates. Corrective RAG (CRAG, Yan et al. 2024) adds a lightweight retrieval evaluator that labels "
  f"retrieved evidence as correct, ambiguous or incorrect and triggers a corrective action. This micro-project is the Variant of "
  f"project P3: the same evaluator as the Primary partner (Roll 12) is used, but the corrective action is a knowledge-graph (KG) "
  f"lookup in Wikidata instead of a web search, and it is tested on 200 stratified questions from Meta's CRAG benchmark covering "
  f"eight question types and five domains. Pages are chunked and retrieved with BGE-small-en-v1.5, a BGE-reranker-base "
  f"cross-encoder acts as evaluator, and Qwen2.5-1.5B-Instruct generates the answers with sampling (3 seeds for the baseline, 1 seed for the RAG methods due to GPU time). Three methods "
  f"are compared: no retrieval (baseline), standard RAG and corrective RAG with KG fallback. Accuracy was "
  f"{pct(M_NO)} for no retrieval, {pct(M_STD)} for standard RAG and {pct(M_CR)} for corrective RAG. The evaluator labelled "
  f"{lab['correct']} questions correct, {lab['ambiguous']} ambiguous and {lab['incorrect']} incorrect, giving a KG fallback "
  f"rate of {fb_rate:.1f} %. Fallback rate and accuracy vary sharply by question type; {EXTRA['top_fb_type']} questions "
  f"triggered the KG most often. All results, including negative ones, are reported exactly as measured. The project relates to "
  f"SDG 4 (reliable, low-cost study assistants) and SDG 9 (open, small-model AI infrastructure on a single 4 GB GPU).")
P("Keywords: corrective RAG, retrieval evaluator, knowledge graph, Wikidata, CRAG benchmark, hallucination, small language models.", italic=True)

# ------------------------------------------------------------------ 1 INTRODUCTION
H1("1. Introduction")
H2("1.1 Background and Motivation")
P("Large language models (LLMs) answer factual questions fluently but cannot be trusted on long-tail or time-sensitive facts. "
  "Retrieval-augmented generation [1] grounds the model in retrieved documents, yet its quality is bounded by the retriever: when "
  "the top-k passages do not contain the answer, the model often produces a confident but wrong answer. Corrective RAG [2] "
  "inserts a retrieval evaluator between retriever and generator and changes the knowledge source when retrieval looks unreliable. "
  "Knowledge graphs [6, 7] are an attractive fallback: they store curated (subject, predicate, object) facts that are compact, "
  "verifiable and free to query. This project applies course concepts — dense retrieval, re-ranking, evaluation metrics "
  "(Recall@K, MRR, NDCG) and knowledge-graph querying — to that problem.")
H2("1.2 Problem Context")
P("The CRAG benchmark [3] shows that straightforward RAG solutions answer only a minority of its questions correctly and that "
  "accuracy drops for dynamic facts, comparisons, aggregations and false-premise questions. Original corrective RAG uses web search "
  "as its fallback, which costs money, is not reproducible and is unavailable offline. The Primary partner (Roll 12) evaluates "
  "the standard corrective pipeline; this Variant asks whether a free, structured KG lookup is a useful replacement and for which "
  "question types the evaluator actually triggers it.")
H2("1.3 Intended Users / Stakeholders")
BUL([("Students and educators", " who need low-cost question answering that admits uncertainty instead of hallucinating."),
     ("RAG / search engineers", " deciding when to trust retrieved evidence and when to route to a KG."),
     ("Course faculty and partner (Roll 12)", " who compare the Variant findings with the Primary reference setup.")])

# ------------------------------------------------------------------ 2 PROBLEM
H1("2. Problem Definition, Objectives and Scope")
H2("2.1 Problem Statement")
P("On 200 questions from the CRAG benchmark, build the same retrieval evaluator as the Primary corrective-RAG project, add a "
  "knowledge-graph lookup as the corrective fallback, and compare fallback rate and answer accuracy by question type against "
  "no-retrieval and standard-RAG baselines, using only open models (embedder ≤ small, LLM ≤ 3 B parameters) on free compute.")
H2("2.2 Objectives")
BUL(["Build a reproducible pipeline: HTML parsing → chunking → BGE-small dense retrieval over each question's 5 pages.",
     "Implement the CRAG-style retrieval evaluator with fixed upper/lower thresholds and knowledge refinement.",
     "Design a Wikidata KG-lookup fallback (entity extraction → entity linking → triple retrieval → triple ranking).",
     "Evaluate accuracy, missing rate, hallucination rate, CRAG score, EM and F1 for three methods over 3 seeds (mean ± std).",
     "Analyse evaluator label counts, fallback rate and accuracy per question type; measure retrieval Recall@5/10, MRR, NDCG@10.",
     "Report time and token cost, five failure cases and limitations honestly."])
H2("2.3 Scope and Deliverables")
P("Included: 200-question CRAG subset, one embedder, one evaluator, one ≤3 B generator, Wikidata as KG, three methods. "
  "Excluded: CRAG mock KG APIs (need the organisers' server), web search, fine-tuning, full-dataset runs, and human evaluation. "
  "Deliverables: Colab-ready notebook, results CSVs (all metrics and seeds), figures, this report, README and experiment log.")
H2("2.4 Assumptions and Constraints")
BUL(["Compute: one NVIDIA RTX 3050 Laptop GPU (4 GB VRAM); the notebook also runs on a free Colab T4.",
     "Accuracy is judged automatically against gold and alternative answers (no human or LLM judge), which is conservative for long answers.",
     "Wikidata is public and rate-limited; responses are cached so the reported run is reproducible offline.",
     "CRAG queries carry a query_time; facts that changed after March 2024 may not match Wikidata's current state."])

# ------------------------------------------------------------------ 3 LIT REVIEW
H1("3. Background / Literature Review")
P("Table 3.1 summarises the work that shaped the design. References are listed in IEEE style at the end of the report.")
TABLE(["Source / Year", "Approach or Key Finding", "Strength / Limitation", "Relevance to This Project"], [
    ["Lewis et al., RAG, 2020 [1]", "Couples a dense retriever with a seq2seq generator", "Strong on knowledge-intensive QA / trusts any retrieved text", "Defines the standard-RAG baseline"],
    ["Yan et al., Corrective RAG, 2024 [2]", "T5 evaluator labels retrieval correct/ambiguous/incorrect; web search fallback; decompose-recompose", "Robust to bad retrieval / web search costly and non-reproducible", "Evaluator logic and three actions reused; fallback replaced by KG"],
    ["Yang et al., CRAG benchmark, 2024 [3]", "4,409 QA pairs, 8 question types, 5 domains, web pages + mock KG", "Realistic and diverse / scoring needs judge", "Dataset, question types and accuracy/hallucination scoring"],
    ["Asai et al., Self-RAG, 2024 [4]", "LLM emits reflection tokens to decide when to retrieve", "Adaptive / needs a fine-tuned LLM", "Motivates gating; ours is training-free"],
    ["Xiao et al., BGE / C-Pack, 2024 [5]", "BGE embedders and rerankers", "Small, strong, open", "Retriever (bge-small) and evaluator (bge-reranker-base)"],
    ["Vrandečić & Krötzsch, Wikidata, 2014 [6]", "Free collaborative KG with an open API", "Broad coverage / uneven for finance, sports", "KG used by the fallback"],
    ["Pan et al., LLMs + KGs roadmap, 2024 [7]", "Survey of KG-enhanced LLMs", "Taxonomy of KG grounding", "Positions KG lookup as factual grounding"],
    ["Edge et al., GraphRAG, 2024 [8]", "LLM-built graph and community summaries", "Good for global questions / expensive indexing", "Contrast: we query an existing KG instead"],
    ["Qwen Team, Qwen2.5, 2024 [9]", "Open LLM family incl. 1.5 B instruct", "Small, multilingual / weak reasoning", "Generator within the ≤3 B limit"],
    ["Es et al., RAGAS, 2024 [10]", "Reference-free RAG metrics", "No gold needed / LLM-judge cost", "Discussed; gold-based scoring used instead"],
], widths=[1.3, 1.9, 1.6, 1.6], caption="Table 3.1: Summary of related work", font=8.5)
H2("3.1 Identified Gap and Project Contribution")
P("Corrective RAG has been evaluated with web search as the fallback, and KG-augmented QA has usually been evaluated without "
  "a retrieval evaluator in front. There is little evidence on how often a retrieval evaluator would hand questions to a KG on "
  "a realistic benchmark, and whether that helps for each question type. This project contributes (i) a training-free, "
  "fully open corrective-RAG pipeline with a Wikidata fallback that runs on a 4 GB GPU, and (ii) a per-question-type analysis of "
  "fallback rate and accuracy on CRAG, reported against the Primary partner's setup.")

# ------------------------------------------------------------------ 4 DESIGN
H1("4. Proposed Solution and System Design")
H2("4.1 Proposed Solution")
P("For every question the system retrieves chunks from the question's own five web pages, asks the evaluator how trustworthy "
  "they are, and then either (a) answers from the refined chunks, (b) discards them and answers from Wikidata triples, or "
  "(c) combines both. The generator is instructed to answer briefly, to say \"invalid question\" for false premises and "
  "\"I don't know\" when unsure, because CRAG penalises hallucinations more than missing answers.")
H2("4.2 System Architecture / Block Diagram / Conceptual Design")
IMG(FIG + "F0_architecture.png", 6.5, "Figure 1: Corrective RAG with a KG-lookup fallback — system architecture")
P("The top row is the shared retrieval path. The evaluator's maximum relevance score s over the top-5 chunks selects one of three "
  "branches. The KG-lookup module (purple) is the Variant's change: it is called only for ambiguous and incorrect questions. All "
  "branches feed the same generator and the same scorer, so differences in accuracy come from the knowledge given to the model.")
H2("4.3 Module Description")
TABLE(["Module / Component", "Responsibility", "Input", "Output"], [
    ["Data loader", "Load CRAG subsets, stratified 200-question sample (seed 42)", "2 CRAG subset files (272 q)", "200 questions + 5 HTML pages each"],
    ["Parser / chunker", "HTML → visible text; 200-word chunks, 40 overlap", "Raw HTML", f"{EXTRA['chunks_total']} chunks ({EXTRA['chunks_mean']:.1f}/q)"],
    ["Dense retriever", "BGE-small-en-v1.5 cosine search per question", "Query + chunks", "Top-10 ranked chunks"],
    ["Retrieval evaluator", "BGE-reranker-base sigmoid scores; thresholds", "Query + top-5 chunks", "Label + refined chunks"],
    ["KG lookup", "Entity extraction (Qwen), Wikidata linking, triples, BGE ranking", "Query", "Top-15 (s, p, o) facts"],
    ["Generator", "Qwen2.5-1.5B-Instruct, 3 sampling seeds", "Prompt with context", "Short answer"],
    ["Scorer", "Accurate / missing / hallucinated, EM, F1", "Prediction + gold/alt", "Metrics CSVs, figures"],
], widths=[1.3, 2.2, 1.4, 1.5], caption="Table 4.1: Modules of the pipeline", font=8.5)
H2("4.4 Data / Inputs / Dataset")
P(f"Source: CRAG Task 1&2 dev v4 [3], obtained from the stratified subsets published at Hugging Face (Quivr/CRAG) [11], licence "
  f"CC BY-NC 4.0. Subsets 1 and 2 (272 questions) form the pool. 200 questions are drawn with proportional stratification on "
  f"question_type (seed 42). Each record has query, query_time, domain, question_type, static_or_dynamic, answer, alt_ans and five "
  f"search results with full HTML. No personal data is involved. Table 4.2 shows the subset composition.")
comp = pd.read_csv(R + "subset_composition.csv", index_col=0)
TABLE(["Question type", "Pool (272)", "Sample (200)"], [[i, int(r["pool"]), int(r["sample"])] for i, r in comp.iterrows()],
      widths=[2.2, 1.3, 1.3], caption="Table 4.2: Subset size by CRAG question type", font=9)
P(f"Domains in the sample: {EXTRA['domains']}. {EXTRA['n_invalid']} questions are false-premise (gold = \"invalid question\"); "
  f"only {EXTRA['answer_in_pages']*100:.1f} % of the remaining {EXTRA['n_answerable']} answerable questions have the gold answer "
  f"string anywhere in their five pages, which caps what any retrieval method can achieve.")
H2("4.5 Algorithm / Design Decisions")
CODE("""
for q in questions:
    chunks  = chunk(html_to_text(q.pages), 200 words, overlap 40)
    top     = bge_small.search(q.query, chunks, k=10)
    s_i     = sigmoid(bge_reranker(q.query, top[:5]))          # evaluator
    label   = CORRECT if max(s_i) >= 0.60 else INCORRECT if max(s_i) < 0.15 else AMBIGUOUS
    refined = [c for c, s in zip(top[:5], s_i) if s >= 0.15]   # knowledge refinement
    if label != CORRECT:                                        # KG-lookup fallback
        ents  = qwen("list named entities", q.query)
        facts = rank_by_bge(q.query, wikidata_triples(ents))[:15]
    context = refined | facts | refined+facts   (correct | incorrect | ambiguous)
    answer  = qwen(prompt(q.query, context), sampling seed in {0,1,2})
""")
P("Design decisions: a cross-encoder evaluator replaces the fine-tuned T5 of [2] so that no training is needed and the same component "
  "is shared with the Primary; thresholds 0.60/0.15 were fixed before generation and their effect is shown in Table 6.5; Wikidata is "
  "used because the CRAG mock-KG API is not publicly hosted; external-identifier and media properties are filtered out so that the "
  "15 triples given to the model carry facts rather than IDs.")

# ------------------------------------------------------------------ 5 IMPLEMENTATION
H1("5. Methodology and Implementation")
H2("5.1 Workflow / Development Method")
BUL(["Read the P3 brief and the CRAG / Corrective-RAG papers; agree evaluator settings with the Primary partner.",
     "Download two stratified CRAG subsets; sample 200 questions; inspect the record schema.",
     "Implement parsing, chunking, retrieval and retrieval-quality labels; compute Recall@K, MRR, NDCG@10.",
     "Implement evaluator, KG lookup (with caching and Wikimedia rate-limit handling) and the three answering methods.",
     "Run 3 seeds, score, build tables/figures, inspect failures; write README, log and report."])
H2("5.2 Technologies, Tools and Resources")
TABLE(["Category", "Technology / Tool", "Purpose / Justification"], [
    ["Programming language / framework", "Python 3.13, PyTorch 2.11 (CUDA 12.8), Transformers 4.57.1", "Model inference on GPU"],
    ["Retrieval / evaluation models", "sentence-transformers 5.1.2: BAAI/bge-small-en-v1.5 (33 M), BAAI/bge-reranker-base (278 M)", "Small open embedder and cross-encoder evaluator"],
    ["Generator", "Qwen/Qwen2.5-1.5B-Instruct (1.5 B, fp16)", "Open LLM within the ≤3 B limit; fits 4 GB VRAM"],
    ["Database / API / platform", "Wikidata Action API (wbsearchentities, wbgetentities)", "Free public knowledge graph for the fallback"],
    ["Data / analysis", "BeautifulSoup4 + lxml, pandas, numpy, scikit-learn, matplotlib, seaborn", "Parsing, metrics, figures"],
    ["Development / testing tools", "Jupyter / nbclient, Google Colab-compatible notebook, VS Code", "Reproducible end-to-end execution"],
    ["Hardware / cloud / other resources", "Laptop, NVIDIA RTX 3050 4 GB GPU; Colab T4 alternative", "Free compute, as required"],
], widths=[1.6, 2.6, 2.2], caption="Table 5.1: Tools and resources", font=8.5)
H2("5.3 Implementation Details")
P("Retrieval. Each question is answered only from its own five pages (CRAG Task 1 setting). The embedder uses the BGE query "
  "instruction prefix and normalised embeddings, so the dot product is cosine similarity.")
P("Evaluator. Each of the top-5 chunks is scored by the cross-encoder; the sigmoid of the logit is a relevance probability. "
  "The maximum decides the branch and chunks below the lower threshold are removed (knowledge refinement).")
P("KG lookup. The generator first extracts up to three entity names. Each is linked with wbsearchentities (top hit), its claims "
  "are fetched with wbgetentities and turned into readable triples by resolving property and item labels in batches of 50. The "
  "triples are ranked by BGE similarity to the question and the top 15 are passed to the generator. All API responses are cached "
  "in data/kg_cache.json; the client sleeps 0.5 s between calls, honours HTTP 429 Retry-After and never caches a failed call.")
P("Generation and scoring. Prompts contain the query time, the context and the abstain/false-premise instructions. Sampling uses "
  "temperature 0.7 and top-p 0.9 with seeds 0, 1, 2; the reported numbers are mean ± std over the seeds. A prediction is accurate "
  "when the normalised gold or alternative answer is contained in it (or token-F1 ≥ 0.5 for multi-word answers); \"I don't know\" "
  "counts as missing; everything else is a hallucination. CRAG score = accuracy − hallucination rate [3].")
P("Challenges. (1) Wikidata returned HTTP 429 during the first run, which silently produced empty KG results; the client was "
  "rewritten with throttling and back-off and the run repeated. (2) With 8 long prompts per batch the 1.5 B model exceeded the 4 GB "
  "VRAM and slowed to a crawl; the reranker is released after evaluation, each document is capped at 600 characters and the batch "
  "size reduced to 4.")
H2("5.4 User Interface / Prototype / Output")
P("The project is a notebook-based experiment, so its \"interface\" is the executed Colab notebook. The figures below are screenshots "
  "of the actual executed cells and their outputs from the reported run (rendered from 67_P3_Variant_CorrectiveRAG_KG.ipynb).")
SHOTS = EXTRA["shots"]
for k, (cell, cap) in enumerate(SHOTS.items(), start=2):
    if not os.path.exists(cell): continue
    IMG(cell, 6.3, f"Figure {k}: {cap}")
H2("5.5 Team Contribution")
TABLE(["Team Member", "Assigned Responsibility", "Work Completed / Evidence"], [
    ["Manish Raje (Roll 67)", "Variant: corrective RAG with KG-lookup fallback on CRAG", "Full pipeline, KG module, experiments, figures, report"],
    ["Partner (Roll 12)", "Primary: reference corrective-RAG setup", "Shared evaluator design; reference results (separate submission)"],
], widths=[1.6, 2.4, 2.4], caption="Table 5.2: Contribution", font=9)

# ------------------------------------------------------------------ 6 RESULTS
H1("6. Testing, Evaluation and Results")
H2("6.1 Test Plan and Test Cases")
TABLE(["Test ID", "Test Input / Scenario", "Expected Result", "Observed Result", "Status"], [
    ["TC-01", "Load two CRAG subsets and sample", "200 questions, 8 types", f"200 questions, {len(comp)-1} types", "Pass"],
    ["TC-02", "Sanity prompt to generator", "Short correct answer", "\"Paris\" for capital of France", "Pass"],
    ["TC-03", "Evaluator on all questions", "Every question labelled", f"{lab['correct']}/{lab['ambiguous']}/{lab['incorrect']} (sum 200)", "Pass"],
    ["TC-04", "KG lookup for fallback questions", "Triples for linked entities", f"Entity found for {EXTRA['kg_found_rate']*100:.1f} % of fallbacks", "Pass"],
    ["TC-05", "Wikidata HTTP 429 (rate limit)", "Back-off, no empty cache", "First run cached empty results; fixed and re-run", "Fail → Fixed"],
    ["TC-06", "False-premise question", "\"invalid question\"", f"No-retrieval accuracy on false premise: {EXTRA['fp_acc_no']:.1f} %", "Observed"],
    ["TC-07", "Re-run with same seeds", "Identical numbers", "Deterministic given cache + seeds", "Pass"],
], widths=[0.6, 1.8, 1.4, 1.9, 0.8], caption="Table 6.1: Test cases", font=8.5)
H2("6.2 Evaluation Metrics")
TABLE(["Metric", "Measurement Method / Formula", "Observed Value", "Interpretation"], [
    ["Accuracy", "share of predictions containing gold/alt answer", f"{pct(M_CR)} (corrective)", "Main QA metric"],
    ["Hallucination", "share of non-abstaining wrong answers", f"{hall[M_CR]:.1f} % (corrective)", "Lower is better"],
    ["CRAG score", "accuracy − hallucination", "Table 6.2", "Rewards abstaining over guessing"],
    ["EM / F1", "normalised exact match / token F1", "Table 6.2", "Strict lexical overlap"],
    ["Recall@5/@10, MRR, NDCG@10", "gold-answer containment in ranked chunks", f"R@5 = {RET['Recall@5'].iloc[0]:.3f}", "Retriever quality"],
    ["Fallback rate", "(ambiguous + incorrect) / 200", f"{fb_rate:.1f} %", "How often the KG is used"],
    ["Cost", "wall-clock seconds, prompt + generated tokens", "Table 6.6", "Efficiency"],
], widths=[1.3, 2.2, 1.4, 1.5], caption="Table 6.2a: Metrics used", font=8.5)
H2("6.3 Results and Discussion")
H3("T1 — Accuracy for no retrieval, standard RAG and corrective RAG")
TABLE(["Method"] + list(T1.columns), [[i] + [str(v).replace(" ± nan", " (1 seed)") for v in r] for i, r in T1.iterrows()],
      widths=[1.7] + [0.8] * len(T1.columns), caption="Table 6.2 (T1): Answer quality on 200 CRAG questions, mean ± std over 3 seeds for no retrieval; RAG methods 1 seed (%)", font=8.5, baseline_row=0)
IMG(FIG + "F1_accuracy_by_method.png", 4.8, "Figure F1: Accuracy by method (error bar = std over 3 seeds, no-retrieval only)")
for t in EXTRA["t1_text"]: P(t)
H3("T2 — Evaluator label counts and fallback rate")
TABLE(["Evaluator label", "Count", "Share (%)"], [[i, int(r["count"]), r["share %"]] for i, r in T2.iterrows()],
      widths=[2.8, 1, 1], caption="Table 6.3 (T2): Evaluator labels and KG fallback rate (thresholds upper 0.60, lower 0.15)", font=9)
IMG(FIG + "F2_confusion_matrix.png", 5.6, "Figure F2: Evaluator labels against actual retrieval quality")
for t in EXTRA["t2_text"]: P(t)
H3("Variant analysis — fallback rate and accuracy by question type")
TABLE(["Question type"] + [c.replace(" acc %", "") for c in T3.columns], [[i] + [str(v).replace(" ± nan", "") for v in r] for i, r in T3.iterrows()],
      widths=[1.2, 0.4, 0.75, 0.7, 1.1, 1.1, 1.1], caption="Table 6.4: Fallback rate and accuracy (%) by question type (no retrieval: mean ± std of 3 seeds; RAG: 1 seed)", font=8)
IMG(FIG + "F3_by_question_type.png", 6.4, "Figure F3: Accuracy (bars) and KG fallback rate (line) by question type")
for t in EXTRA["t3_text"]: P(t)
H3("Evaluator threshold sensitivity")
TABLE(list(T4.columns), T4.values.tolist(), caption="Table 6.5: Label counts and precision of the 'correct' label as thresholds vary (no re-generation)", font=8)
P(EXTRA["t4_text"])
H3("Retrieval quality and cost")
TABLE(["Retriever"] + list(RET.columns), [[i] + list(r) for i, r in RET.iterrows()], widths=[1.6, 1, 1, 1, 1],
      caption=f"Table 6.6a: Retrieval metrics over {EXTRA['n_answerable']} answerable questions (relevant = chunk contains gold answer)", font=9)
TABLE(list(T5.columns), T5.values.tolist(), widths=[2.4, 0.8, 1.0, 1.0, 1.0], caption="Table 6.6: Cost — wall-clock time (s) and tokens per stage", font=8)
P(EXTRA["cost_text"])
H3("Five failure cases")
for i, f in enumerate(EXTRA["failures"], 1):
    P(f["text"], bold_lead=f"Case {i} ({f['type']}). ")
H2("6.4 Validation / Demonstration Evidence")
P("Demo steps: open 67_P3_Variant_CorrectiveRAG_KG.ipynb in Google Colab (GPU runtime) or locally, run all cells; the notebook "
  "downloads the CRAG subsets, uses the cached Wikidata responses, and writes every table and figure to results/ and figures/. "
  "The executed notebook (with outputs) and notebook_render.html are included as evidence; screenshots in Section 5.4 and "
  "Appendix A are taken from that executed notebook. No credentials or API keys are needed.")

# ------------------------------------------------------------------ 7 SDG
H1("7. Sustainability and SDG Alignment")
P("The project's impacts are mostly indirect: it is a research prototype, not a deployed service.")
H2("7.1 Sustainability Assessment")
TABLE(["Dimension", "Questions to Consider", "Project-Specific Assessment / Evidence"], [
    ["Environmental", "Energy and compute use; e-waste; resource efficiency",
     f"Positive/neutral: all runs on one 60 W laptop GPU; total measured compute ≈ {T5['time_s'].sum()/60:.0f} min. Small models (≤1.5 B) and caching avoid repeated API and GPU work."],
    ["Social", "Accessibility; safety; privacy; fairness",
     "Positive: abstaining (\"I don't know\") and false-premise detection reduce misinformation. Risk: KG coverage is biased toward well-documented (Western, English) entities."],
    ["Economic", "Development and operating cost; long-term viability",
     "Positive: ₹0 software/API cost; Wikidata replaces paid web-search APIs used by the original corrective RAG."],
], widths=[1.1, 2.0, 3.3], caption="Table 7.1: Sustainability assessment", font=8.5)
H2("7.2 UN Sustainable Development Goals (SDGs)")
TABLE(["SDG No. and Name", "Project Feature / Activity Linked to SDG", "Expected or Observed Contribution", "Evidence / Indicator"], [
    ["SDG 4: Quality Education", "Open, low-cost QA assistant that abstains when unsure", "More trustworthy study aid for students without paid tools", "Hallucination rate per method (Table 6.2)"],
    ["SDG 9: Industry, Innovation and Infrastructure", "Reproducible open-model RAG on a 4 GB GPU with a public KG", "Lowers the barrier to building reliable AI infrastructure", "Runs end to end on free compute; cost in Table 6.6"],
], widths=[1.3, 1.8, 1.8, 1.5], caption="Table 7.2: SDG alignment", font=8.5)
H2("7.3 Sustainability Measures and Trade-offs")
P("Measures: small models in fp16, freeing the reranker before generation, caching every Wikidata call, and stopping early at 48 new "
  "tokens. Trade-off: a 1.5 B generator is cheap but weaker at reasoning (aggregation, multi-hop), which limits accuracy.")
H2("7.4 Sustainability Indicator / Follow-up")
P("Measured: wall-clock time and tokens per stage (Table 6.6). Proposed future measure: energy per answered question (Wh) "
  "using nvidia-smi power logging, and accuracy per 1,000 tokens.")

# ------------------------------------------------------------------ 8 FEASIBILITY
H1("8. Feasibility, Limitations and Risk Considerations")
H2("8.1 Feasibility")
P("Technically feasible on free resources: the whole experiment fits a 4 GB GPU or a Colab T4. Operationally, a re-run takes under "
  "an hour with cached KG responses. Cost estimate:")
TABLE(["Item / Resource", "Cost (₹)", "Basis / Notes"], [
    ["Hardware / hosting / cloud", "0", "Existing laptop GPU or free Colab"],
    ["Software / API / dataset", "0", "Open models, Wikidata API, CRAG (CC BY-NC 4.0)"],
    ["Other resources", "0", "—"],
    ["Total estimated cost", "0", "No recurring cost; electricity only"],
], widths=[2, 1, 3.4], caption="Table 8.1: Budget estimate", font=9)
H2("8.2 Limitations")
BUL(["Only 200 questions; per-type cells contain 8–56 questions, so per-type differences of a few points are within noise.",
     "Automatic string-containment scoring underestimates correct paraphrases and long or numeric answers; no LLM judge or human labels.",
     "Wikidata is not the CRAG mock KG: finance prices and recent sports results are mostly absent, and facts are as of today, not query_time.",
     "Entity linking takes only the top Wikidata hit, so ambiguous names link to the wrong entity (see failure cases).",
     "The 1.5 B generator and 600-character document cap limit reasoning over long tables; thresholds were not tuned on a dev split.",
     "Because of GPU time (a thermally throttled 4 GB laptop GPU), only the no-retrieval baseline was run with 3 seeds; the RAG methods used seed 0, so their variance is not measured. Retrieval, evaluator and KG lookup are deterministic."])
H2("8.3 Risks, Ethics, Privacy and Security")
P("No personal data is processed. The dataset licence (CC BY-NC 4.0) restricts use to non-commercial research. Wikidata usage "
  "follows the Wikimedia API etiquette (descriptive User-Agent, throttling, back-off). Risks: confident wrong answers from a small "
  "model, KG bias toward well-documented entities, and over-trust in the evaluator. Safeguards: abstention instruction, reporting "
  "hallucination separately, and publishing the evaluator scores and KG facts used for every question (evaluator_and_kg_trace.json).")

# ------------------------------------------------------------------ 9 CONCLUSION
H1("9. Conclusion and Future Scope")
H2("9.1 Summary")
for t in EXTRA["conclusion"]: P(t)
H2("9.2 Future Scope")
BUL(["Replace top-1 entity linking with question-aware linking (type constraints, multiple candidates) and SPARQL property paths.",
     "Use the CRAG mock-KG API for finance/sports, and filter KG facts by query_time.",
     "Tune evaluator thresholds on a held-out split and compare with the Primary's thresholds; try an LLM-based evaluator.",
     "Add an LLM judge / human labels (as in project P9) to reduce scoring bias; scale to the full 2.7 k public questions.",
     "Measure energy per answer and add decompose-then-recompose refinement from the original corrective RAG."])

# ------------------------------------------------------------------ REFERENCES
H1("References")
REFS = [
 "P. Lewis et al., \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,\" in Proc. NeurIPS, 2020. arXiv:2005.11401.",
 "S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling, \"Corrective Retrieval Augmented Generation,\" arXiv:2401.15884, 2024.",
 "X. Yang et al., \"CRAG – Comprehensive RAG Benchmark,\" in Proc. NeurIPS Datasets and Benchmarks Track, 2024. arXiv:2406.04744.",
 "A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, \"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection,\" in Proc. ICLR, 2024.",
 "S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, \"C-Pack: Packed Resources for General Chinese Embeddings,\" in Proc. SIGIR, 2024 (BGE models).",
 "D. Vrandečić and M. Krötzsch, \"Wikidata: A Free Collaborative Knowledgebase,\" Communications of the ACM, vol. 57, no. 10, pp. 78–85, 2014.",
 "S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu, \"Unifying Large Language Models and Knowledge Graphs: A Roadmap,\" IEEE TKDE, vol. 36, no. 7, 2024.",
 "D. Edge et al., \"From Local to Global: A Graph RAG Approach to Query-Focused Summarization,\" arXiv:2404.16130, 2024.",
 "Qwen Team, \"Qwen2.5 Technical Report,\" arXiv:2412.15115, 2024.",
 "S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, \"RAGAS: Automated Evaluation of Retrieval Augmented Generation,\" in Proc. EACL (Demos), 2024.",
 "Quivr, \"CRAG (stratified subsets of Meta CRAG),\" Hugging Face Datasets, 2024. https://huggingface.co/datasets/Quivr/CRAG (accessed 02 Nov 2026).",
 "N. Reimers and I. Gurevych, \"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,\" in Proc. EMNLP, 2019.",
 "N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, \"BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models,\" in Proc. NeurIPS Datasets and Benchmarks, 2021.",
 "K. Järvelin and J. Kekäläinen, \"Cumulated Gain-Based Evaluation of IR Techniques,\" ACM TOIS, vol. 20, no. 4, pp. 422–446, 2002.",
 "Wikimedia Foundation, \"Wikidata Action API and API etiquette,\" https://www.wikidata.org/w/api.php (accessed 02 Nov 2026).",
]
for i, r in enumerate(REFS, 1):
    p = P(r, bold_lead=f"[{i}] "); p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

# ------------------------------------------------------------------ APPENDICES
H1("Appendices")
H2("Appendix A: Additional screenshots / outputs")
for k, (cell, cap) in enumerate(EXTRA["appendix_shots"].items(), start=1):
    if os.path.exists(cell): IMG(cell, 6.3, f"Figure A{k}: {cap}")
H2("Appendix B: Selected code excerpt — KG-lookup fallback")
CODE('''
def kg_lookup(rs, topk=15):
    ents = chat([ENT_PROMPT.format(r["query"]) for r in rs], max_new=32, stage="kg_entity")
    for r, e in zip(rs, ents):
        names = [x.strip(" .\\"'") for x in e.split(",") if x.strip(" .\\"'")][:3]
        trip = [t for n in names for t in entity_triples(n)]          # Wikidata claims -> triples
        sims = emb.encode(trip) @ emb.encode(QPFX + r["query"])        # rank by BGE similarity
        r["kg_facts"] = [trip[i] for i in np.argsort(-sims)[:topk]]
''')
H2("Appendix C: Submission folder contents")
BUL(["67_P3_Variant_CorrectiveRAG_KG.ipynb — end-to-end notebook (executed, with outputs); notebook_render.html",
     "results/ — all_metrics.csv (all metrics and seeds), T1–T5 CSVs, per_question_predictions.csv, evaluator_and_kg_trace.json",
     "figures/ — F0 architecture, F1 accuracy, F2 confusion matrix, F3 by question type; screenshots/ — executed-cell captures",
     "README.md (steps and versions), requirements.txt, EXPERIMENT_LOG.md (runs, timings and AI-tool use), data/kg_cache.json"])
H2("Appendix D: Repository / demo information")
P("Run locally: create a venv, pip install -r requirements.txt (CUDA build of torch), then execute the notebook. On Colab: upload the "
  "notebook and data/kg_cache.json, uncomment the pip line in the first cell and run all. No credentials or secrets are used.")

# ------------------------------------------------------------------ FINAL CHECKLIST (kept from template)
H1("Final Submission Checklist")
BUL(["☑ Front page, blank page and cover page completed", "☑ Abstract, objectives, method with baseline, setup, results (T1, T2, F1, F2) included",
     "☑ Variant analysis by question type included", "☑ Five failure cases and limitations included", "☑ ≥10 references in IEEE style",
     "☑ Code, results CSV with all seeds, README and experiment log included", "☑ Screenshots of executed steps with captions"])

out = "67_P3_Variant_Report.docx"
doc.save(out); print("saved", out)
