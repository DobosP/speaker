from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


class DuplicateScanner:
    def __init__(self, root: str | Path = "."):
        self.root = Path(root)

    def scan(self) -> dict[str, object]:
        return {
            "test_functions": self._scan_test_functions(),
            "failure_corpus": self._scan_failure_corpus(),
            "audio_corpora": self._scan_audio_corpora(),
            "ownership_hints": self._ownership_hints(),
        }

    def write(self, path: Path) -> dict[str, object]:
        report = self.scan()
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _scan_test_functions(self) -> dict[str, object]:
        occurrences: dict[str, list[str]] = defaultdict(list)
        for test_file in sorted((self.root / "tests").glob("test_*.py")):
            try:
                tree = ast.parse(test_file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                    occurrences[node.name].append(str(test_file))
                elif isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef) and child.name.startswith("test_"):
                            occurrences[f"{node.name}.{child.name}"].append(str(test_file))
        duplicates = {
            name: paths for name, paths in occurrences.items() if len(set(paths)) > 1
        }
        repeated_short_names = defaultdict(list)
        for name, paths in occurrences.items():
            short = name.split(".")[-1]
            repeated_short_names[short].extend(paths)
        repeated_short_names = {
            name: sorted(set(paths))
            for name, paths in repeated_short_names.items()
            if len(set(paths)) > 1
        }
        return {
            "total_test_names": len(occurrences),
            "duplicate_full_names": duplicates,
            "duplicate_short_names": repeated_short_names,
        }

    def _scan_failure_corpus(self) -> dict[str, object]:
        return self._scan_corpus(self.root / "tests" / "fixture_audio" / "failure_discovery")

    def _scan_audio_corpora(self) -> dict[str, object]:
        fixture_root = self.root / "tests" / "fixture_audio"
        corpora = {}
        for corpus in sorted(fixture_root.glob("*")):
            if corpus.is_dir() and (corpus / "metadata.json").exists():
                corpora[corpus.name] = self._scan_corpus(corpus)
        return corpora

    def _scan_corpus(self, corpus: Path) -> dict[str, object]:
        metadata_path = corpus / "metadata.json"
        if not metadata_path.exists():
            return {"present": False}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cases = metadata.get("cases", [])
        hashes: dict[str, list[str]] = defaultdict(list)
        for sample in sorted(corpus.glob("*.npy")):
            hashes[hashlib.sha256(sample.read_bytes()).hexdigest()].append(sample.name)
        duplicate_hashes = [names for names in hashes.values() if len(names) > 1]
        return {
            "present": True,
            "path": str(corpus),
            "case_count": len(cases),
            "sample_count": len(list(corpus.glob("*.npy"))),
            "duplicate_names": self._duplicates([case.get("name", "") for case in cases]),
            "duplicate_descriptions": self._duplicates([case.get("description", "") for case in cases]),
            "duplicate_importance": self._duplicates([case.get("importance", "") for case in cases]),
            "duplicate_audio_hash_groups": duplicate_hashes,
        }

    def _ownership_hints(self) -> dict[str, object]:
        keywords = ("barge", "echo", "vad", "stt", "tts", "llm", "profile", "record", "wakeword")
        result: dict[str, list[str]] = {keyword: [] for keyword in keywords}
        for test_file in sorted((self.root / "tests").glob("test_*.py")):
            text = test_file.read_text(encoding="utf-8", errors="ignore").lower()
            for keyword in keywords:
                if keyword in text:
                    result[keyword].append(str(test_file))
        return {
            key: paths for key, paths in result.items() if len(paths) > 1
        }

    def _duplicates(self, values: list[str]) -> dict[str, int]:
        counts = Counter(value for value in values if value)
        return {value: count for value, count in counts.items() if count > 1}
