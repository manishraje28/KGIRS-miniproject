"""Render the executed notebook to HTML and screenshot selected cells (input + real output)."""
import sys, subprocess, os
from playwright.sync_api import sync_playwright
NB = "67_P3_Variant_CorrectiveRAG_KG.ipynb"
subprocess.run([r".venv\Scripts\python", "-m", "jupyter", "nbconvert", "--to", "html", NB, "--output", "notebook_render"], check=True)
os.makedirs("screenshots", exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=1.5)
    pg.goto("file:///" + os.path.abspath("notebook_render.html").replace("\\", "/")); pg.wait_for_timeout(1500)
    cells = pg.query_selector_all(".jp-Cell.jp-CodeCell")
    print(len(cells), "code cells")
    for i, c in enumerate(cells):
        c.screenshot(path=f"screenshots/cell_{i:02d}.png")
    b.close()
