import json
from pathlib import Path
from typing import Any

def read_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(content: str, path: str | Path):
    with open(path, "w") as f:
        json.dump(content, f)
