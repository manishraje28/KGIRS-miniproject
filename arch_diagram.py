"""Figure 1: system architecture of corrective RAG with a KG-lookup fallback."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(11, 5.4)); ax.set_xlim(0, 110); ax.set_ylim(0, 54); ax.axis("off")
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=c, ec="#333", lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.6)
def arr(x1, y1, x2, y2, t=""):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333"))
    if t: ax.text((x1 + x2) / 2 + 0.6, (y1 + y2) / 2 + 0.8, t, fontsize=7.8, color="#a33", style="italic")
box(1, 40, 15, 9, "CRAG question\n(200, 8 types)\n+ 5 web pages", "#fde9c9")
box(21, 40, 17, 9, "HTML → text\n200-word chunks\n(40 overlap)", "#e3eefa")
box(43, 40, 17, 9, "BGE-small-en-v1.5\ndense retrieval\ntop-10 chunks", "#e3eefa")
box(65, 40, 20, 9, "Retrieval evaluator\nBGE-reranker-base\nmax sigmoid score s", "#f7d9d9")
arr(16.8, 44.5, 20.2, 44.5); arr(38.8, 44.5, 42.2, 44.5); arr(60.8, 44.5, 64.2, 44.5)
box(66, 26, 18, 7, "CORRECT  (s ≥ 0.60)\nknowledge refinement", "#d8f0d8")
box(40, 26, 20, 7, "AMBIGUOUS (0.15 ≤ s < 0.60)\nrefined chunks + KG facts", "#fff4c2")
box(88, 26, 20, 7, "INCORRECT (s < 0.15)\ndiscard web → KG only", "#f7d9d9")
arr(75, 39.5, 75, 33.8); arr(70, 39.5, 52, 33.8); arr(80, 39.5, 96, 33.8)
box(2, 22, 32, 14, "KG-lookup fallback (Variant)\n1. Qwen extracts entities\n2. Wikidata wbsearchentities → QID\n3. claims → (s, p, o) triples\n4. BGE ranks triples, keep top-15", "#e8dcf5")
arr(39.5, 29.5, 34.5, 29.5); ax.text(34.8, 31, "fallback", fontsize=7.8, color="#a33", style="italic"); ax.annotate("", (34.5, 33), (98, 33.8), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#7a4fa8", connectionstyle="arc3,rad=0.25", ls="--"))
box(30, 6, 32, 9, "Generator: Qwen2.5-1.5B-Instruct (fp16)\nprompt = question + context\n3 sampling seeds", "#e3eefa")
arr(18, 21.5, 40, 15.6); arr(50, 25.5, 46, 15.6); arr(75, 25.5, 56, 15.6); arr(98, 25.5, 60, 14)
box(70, 6, 30, 9, "Scoring vs gold / alt answers\naccuracy · missing · hallucination\nEM · F1 · per question type", "#fde9c9")
arr(62.8, 10.5, 69.2, 10.5)
ax.text(1, 52, "Baselines: (a) no retrieval — question only;  (b) standard RAG — always top-5 chunks, no evaluator", fontsize=8.5)
fig.tight_layout(); fig.savefig("figures/F0_architecture.png", dpi=170)
