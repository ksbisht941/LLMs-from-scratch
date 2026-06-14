import json

notebook_path = "llm-fine-tuning/4-preferenced-fine-tuning.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

for idx, cell in enumerate(notebook.get("cells", [])):
    source_str = "".join(cell.get('source', []))[:120].replace('\n', ' ')
    print(f"Cell {idx} ({cell.get('cell_type')}): {source_str}")
