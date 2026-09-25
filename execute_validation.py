"""Execute the code-validation notebook and preserve actual Jupyter outputs."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.cache' / 'jupyter')
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True, exist_ok=True)
path = root / 'notebooks' / '00_validation.ipynb'
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
NotebookClient(nb, timeout=180, resources={'metadata': {'path': str(root)}}).execute()
nbformat.write(nb, path)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
print('Saved notebook with real execution outputs:', path.name)
