# Experiment Log — Roll 67, P3 Corrective RAG (Variant)

| Date / time (IST) | What was run | Outcome |
|---|---|---|
| 02-11-2026 21:05 | Inspected front page and department report template | Report layout planned |
| 21:07 | Downloaded CRAG subsets 1–2 (Quivr/CRAG, 272 q) | Parsed; 8 question types confirmed |
| 21:10 | Created venv, CUDA torch 2.11 (cu128), transformers 4.57.1 | GPU (RTX 3050 4 GB) available |
| 21:17 | First notebook run | Stalled: LLM + reranker exceeded 4 GB VRAM; reranker now freed before generation |
| 21:55 | Second run | Wikidata returned HTTP 429; failures were cached as empty. Rewrote client (throttle 0.5 s, Retry-After, no caching of failures) |
| 22:08 | KG lookup run as script | 35 fallback questions, entity found for 82.9 %; cache saved to data/kg_cache.json |
| 22:15 | Full run, 3 seeds × 3 methods, 8×1200-char docs | Too slow on thermally throttled GPU (~56/200 answers in 40 min); stopped |
| 23:05 | Final run: docs capped at 400 chars, batch 4; baseline 3 seeds, RAG methods seed 0 | Completed; all results in results/ |
| 23:35 | analysis.py, shots.py, build_report.py, Word → PDF | Report generated |

**Deviation from brief:** the RAG methods were run with one seed (seed 0) instead of three because of GPU time; this is stated in the report.

## AI tool use
An AI coding assistant was used to write the pipeline, notebook, analysis and report-building scripts, and to draft report text. All numbers in the report are produced by the executed notebook and read programmatically from results/*.csv; none were typed by hand.
