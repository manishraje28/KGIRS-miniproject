"""Simple SDG badges (number + name in the UN goal colour) for the cover page."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
goals = [(4, "QUALITY\nEDUCATION", "#C5192D"), (9, "INDUSTRY, INNOVATION\nAND INFRASTRUCTURE", "#FD6925")]
fig, ax = plt.subplots(figsize=(4.2, 1.6)); ax.set_xlim(0, 4.2); ax.set_ylim(0, 1.6); ax.axis("off")
for i, (n, name, c) in enumerate(goals):
    x = 0.1 + i * 2.1
    ax.add_patch(FancyBboxPatch((x, 0.1), 1.9, 1.4, boxstyle="round,pad=0,rounding_size=0.08", fc=c, ec="none"))
    ax.text(x + 0.12, 1.2, str(n), color="white", fontsize=26, fontweight="bold", va="center")
    ax.text(x + 0.62, 1.2, "SDG", color="white", fontsize=10, fontweight="bold", va="center")
    ax.text(x + 0.12, 0.5, name, color="white", fontsize=8.5, fontweight="bold", va="center")
fig.savefig("figures/sdg_badges.png", dpi=200, bbox_inches="tight", transparent=True)
