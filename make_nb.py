"""Convert notebook_src.py (# %% cells) into a Colab-ready .ipynb."""
import re, nbformat as nbf
src = open("notebook_src.py", encoding="utf8").read()
nb = nbf.v4.new_notebook(); nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    if block.startswith(" [markdown]"):
        body = block.split("\n", 1)[1]
        nb.cells.append(nbf.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().splitlines())))
    else:
        nb.cells.append(nbf.v4.new_code_cell(block.split("\n", 1)[1].strip()))
nbf.write(nb, "67_P3_Variant_CorrectiveRAG_KG.ipynb"); print(len(nb.cells), "cells")
