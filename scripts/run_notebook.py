"""Execute the notebook and preserve cell outputs even when execution fails."""
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MPLBACKEND', 'module://matplotlib_inline.backend_inline')
runtime = ROOT / 'incremental_outputs' / 'jupyter_runtime'
runtime.mkdir(parents=True, exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
notebook_path = ROOT / 'dni_incremental_learning_framework.ipynb'
nb = nbformat.read(notebook_path, as_version=4)


def on_start(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'{time.strftime("%H:%M:%S")} Starting cell {cell_index}', flush=True)


def on_executed(cell, cell_index, **kwargs):
    nbformat.write(nb, notebook_path)
    print(f'{time.strftime("%H:%M:%S")} Finished cell {cell_index}', flush=True)
    if cell_index == 25:
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text, flush=True)


client = NotebookClient(nb, timeout=None, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT)}},
                        on_cell_execute=on_start, on_cell_executed=on_executed)
try:
    client.execute()
finally:
    nbformat.write(nb, notebook_path)
print('Notebook execution completed.', flush=True)
