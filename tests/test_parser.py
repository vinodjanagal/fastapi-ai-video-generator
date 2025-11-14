# tests/test_parser.py
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.engine.parser import semantic_parser

prompts_to_test = [
    "an inventor studying a glowing holographic schematic inside a warm wooden cabin at dawn",
    "a chef slicing vibrant vegetables in a modern kitchen",
    "a knight inspecting a runestone in a dark forest",
    "a cat on a windowsill"
]

print("--- Running Parser Unit Tests ---")
for p in prompts_to_test:
    print(f"\n[PROMPT]: {p}")
    parts = semantic_parser(p)
    print(f"[RESULT]: {parts}")
    assert "subject" in parts, "Parser failed to find a subject."
print("\n--- ✅ All Parser Unit Tests Passed ---")