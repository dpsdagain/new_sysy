import json
import os
import uuid
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class NotebookMutator:
    """
    Anthropic-Grade Jupyter Mutator (F-12).
    Surgically edits .ipynb cells without corrupting JSON.
    """
    
    @staticmethod
    def edit(file_path: str, cell_id: str, new_source: str, 
             edit_mode: str = "replace", cell_type: str = "code") -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            cells: List[Dict[str, Any]] = nb.get("cells", [])
            target_idx = -1

            # 🧬 Cell Discovery (mirrors NotebookEditTool.ts:354)
            # Try UUID first, then fall back to virtual index (cell-0 style)
            for i, cell in enumerate(cells):
                if cell.get("id") == cell_id:
                    target_idx = i
                    break
            
            if target_idx == -1 and cell_id.startswith("cell-"):
                try:
                    target_idx = int(cell_id.split("-")[1])
                except:
                    pass

            if edit_mode != "insert" and (target_idx < 0 or target_idx >= len(cells)):
                return f"Error: Cell ID '{cell_id}' not found."

            # 🛠️ Mutation Logic (NotebookEditTool.ts:392-428)
            if edit_mode == "delete":
                cells.pop(target_idx)
            elif edit_mode == "insert":
                new_cell = {
                    "cell_type": cell_type,
                    "metadata": {},
                    "source": new_source.splitlines(keepends=True)
                }
                if cell_type == "code":
                    new_cell.update({"execution_count": None, "outputs": []})
                
                # Insert after (if replacing past end) or at index
                insert_pos = target_idx + 1 if target_idx != -1 else 0
                cells.insert(insert_pos, new_cell)
            else: # replace
                target = cells[target_idx]
                target["source"] = new_source.splitlines(keepends=True)
                if target["cell_type"] == "code":
                    # 🧹 Sanitize stales (NotebookEditTool.ts:422)
                    target["execution_count"] = None
                    target["outputs"] = []

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1) # Anthropic Indent (Line 431)

            return f"Successfully {edit_mode}ed cell in {file_path}."
        except Exception as e:
            return f"Error mutating notebook: {str(e)}"
