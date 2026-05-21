"""SHA-256 keyed JSONL cache for LLM responses."""

import json
import hashlib
from pathlib import Path
from typing import Optional


class ResponseCache:

    def __init__(self, path: str = "results/llm_correction/responses.jsonl"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict = {}
        self._load()

    def _load(self):
        if self._path.exists():
            with open(self._path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        self._store[entry["key"]] = entry["value"]
                    except Exception:
                        pass

    @staticmethod
    def make_key(text: str, model: str, prompt_strategy: str) -> str:
        raw = f"{text}|{model}|{prompt_strategy}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def put(self, key: str, value: str):
        self._store[key] = value
        with open(self._path, "a") as f:
            f.write(json.dumps({"key": key, "value": value}) + "\n")
