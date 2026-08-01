#!/usr/bin/env python
"""
Refactor GUI imports from 'from gui.pyptv.X import Y' to 'from pyptv.X import Y'

This makes the code work both when:
1. Running from source (gui/ is in sys.path)
2. Installed as a package (pyptv is the importable module)
"""

import re
from pathlib import Path


def refactor_imports(file_path):
    """Replace gui.pyptv imports with relative imports."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # Pattern 1: from gui.pyptv.X import Y -> from .X import Y (relative)
    # Pattern 2: from gui.pyptv import X -> from . import X (relative)
    # Pattern 3: import gui.pyptv.X -> not common, skip for now

    # Replace 'from gui.pyptv.' with 'from .'
    content = re.sub(
        r"from gui\.pyptv\.([a-zA-Z_][a-zA-Z0-9_]*) import", r"from .\1 import", content
    )

    # Replace 'from gui.pyptv import' with 'from . import'
    content = re.sub(r"from gui\.pyptv import", r"from . import", content)

    if content != original:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def main():
    gui_pyptv_dir = Path(__file__).parent / "gui" / "pyptv"

    changed_files = []
    for py_file in gui_pyptv_dir.glob("*.py"):
        if refactor_imports(py_file):
            changed_files.append(py_file.name)
            print(f"✅ Refactored: {py_file.name}")

    print(f"\nRefactored {len(changed_files)} files")
    print("Files changed:", ", ".join(changed_files))


if __name__ == "__main__":
    main()
