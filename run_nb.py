"""Execute the notebook cell by cell, saving after each cell and logging progress."""
import sys, time, nbformat
from nbclient import NotebookClient
path = sys.argv[1]; nb = nbformat.read(path, as_version=4)
c = NotebookClient(nb, timeout=None, kernel_name="kgirs67", resources={"metadata": {"path": "."}})
with c.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code": continue
        t = time.time(); c.execute_cell(cell, i)
        print(f"cell {i} done in {time.time()-t:.1f}s", flush=True); nbformat.write(nb, path)
print("ALL DONE", flush=True)
