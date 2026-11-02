# 67_P3_Variant — Corrective RAG with a KG-Lookup Fallback on CRAG

**Student:** Manish Raje · Roll 67 · T.E. COMP D17B · KGIRS Lab (26CMPE742) · Partner (Primary): Roll 12

On 200 stratified questions from the CRAG benchmark, the same retrieval evaluator as the Primary project is used,
but the corrective action is a **Wikidata knowledge-graph lookup**. The project compares no retrieval (baseline),
standard RAG and corrective RAG, and reports fallback rate and accuracy **by question type**.

## Folder
| Path | Content |
|---|---|
| `67_P3_Variant_CorrectiveRAG_KG.ipynb` | End-to-end notebook (executed, with outputs). Colab-ready. |
| `notebook_render.html` | HTML render of the executed notebook |
| `67_P3_Variant_Report.docx` / `.pdf` | Report (front page, blank page, department template) |
| `results/all_metrics.csv` | **All metrics for all methods and seeds** (+ retrieval metrics) |
| `results/T1…T5_*.csv`, `per_question_predictions.csv`, `evaluator_and_kg_trace.json` | Tables, per-question outputs, evaluator scores and KG facts |
| `figures/` | F0 architecture, F1 accuracy, F2 confusion matrix, F3 by question type |
| `screenshots/` | Screenshots of every executed code cell |
| `data/kg_cache.json` | Cached Wikidata responses (makes the run reproducible offline) |
| `EXPERIMENT_LOG.md` | What was run, when, issues, and AI-tool use |
| `notebook_src.py`, `make_nb.py`, `run_nb.py`, `analysis.py`, `build_report.py`, `shots.py`, `arch_diagram.py`, `sdg_badges.py` | Source of the notebook and report build |

## Steps to reproduce
1. **Colab:** upload the notebook and `data/kg_cache.json` (into `data/`), choose a GPU runtime, uncomment the `pip` line in the first cell, then Run all.
2. **Local (Windows/Linux, CUDA GPU ≥ 4 GB):**
   ```bash
   python -m venv .venv
   .venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
   .venv/Scripts/pip install -r requirements.txt
   .venv/Scripts/python -m ipykernel install --user --name kgirs67
   .venv/Scripts/python run_nb.py 67_P3_Variant_CorrectiveRAG_KG.ipynb
   ```
3. Rebuild the report: `python analysis.py && python shots.py && python build_report.py`.

The notebook downloads CRAG subsets 1 and 2 from `huggingface.co/datasets/Quivr/CRAG` automatically.

## Versions
Python 3.13.6 · torch 2.11.0+cu128 · transformers 4.57.1 · sentence-transformers 5.1.2 · see `requirements.txt`.
Models: `BAAI/bge-small-en-v1.5` (embedder), `BAAI/bge-reranker-base` (evaluator), `Qwen/Qwen2.5-1.5B-Instruct` (generator).
Seeds: sampling seed 42, generation seeds 0/1/2. Hardware: NVIDIA RTX 3050 Laptop 4 GB.
