"""Strict validation, ID-based joins and Unicode tokenization."""
import json
import re
from pathlib import Path


def tokens(text):
    # Preserve Russian letters AND numbers: dates, amounts and policy limits matter.
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def records(path, field, aliases=()):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict) and Path(path).stem in raw:
        raw = raw[Path(path).stem]
    if isinstance(raw, dict):
        raw = [dict(value, id=key) if isinstance(value, dict)
               else {"id": key, field: value} for key, value in raw.items()]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected nonempty list or ID mapping")
    result = []
    for i, item in enumerate(raw):
        row = {"id": str(i), field: item} if isinstance(item, str) else dict(item)
        row["id"] = str(row.get("id", row.get("question_id", i)))
        value = next((row[k] for k in (field, *aliases) if k in row), None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: invalid {field}, row {i}")
        row[field] = value.strip()
        result.append(row)
    ids = [row["id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate IDs")
    return result


def load_dataset(config):
    base = Path(config["data_dir"])
    articles = records(base / "articles.json", "text", ("content", "body", "article"))
    questions = records(base / "questions.json", "question", ("text",))
    answers = records(base / "ground_truth.json", "answer", ("ground_truth", "text", "reference"))
    refs = {r["id"]: r["answer"] for r in answers}
    if set(refs) != {q["id"] for q in questions}:
        raise ValueError("Question IDs and reference IDs do not match")
    for name, rows in (("articles", articles), ("questions", questions)):
        if len(rows) != config[f"expected_{name}"]:
            raise ValueError(f"Expected {config[f'expected_{name}']} {name}; got {len(rows)}")
    return articles, questions, refs
