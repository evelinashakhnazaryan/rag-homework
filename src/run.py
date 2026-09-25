"""Run each experiment separately; completed answers survive interruption."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from statistics import mean
from .data import load_dataset


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(config, mode):
    articles, questions, refs = load_dataset(config)
    from .evaluation import lexical_metrics, judge_answer
    from .models import groq, LocalModel, messages
    output = Path(config["results_dir"])
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(
        [config, articles, questions, refs], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest = {"fingerprint": fingerprint, "config": config, "python": platform.python_version(),
                "packages": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Data/config changed: select a new results_dir")
    else:
        write_json(manifest_path, manifest)
    path = output / f"{mode}.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    retriever = None
    if mode == "groq_rag":
        from .retrieval import build_index
        store, retriever, chunks = build_index(articles, config)
        store.dump(str(output / "index.json"))
        write_json(output / "chunks.json", [{"text": c.page_content, **c.metadata} for c in chunks])
        print(f"Index: {len(articles)} articles, {len(chunks)} chunks", flush=True)
    model = LocalModel(config) if mode == "local_zero" else groq(config)
    judge = groq(config, judge=True)
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or not set(by_id).issubset(refs):
        raise ValueError("Invalid checkpoint IDs")
    for question in questions:
        qid = question["id"]
        row = by_id.get(qid)
        if row is None:
            docs = retriever.invoke(question["question"]) if retriever else None
            start = time.perf_counter()
            answer = (model.answer(question["question"]) if mode == "local_zero" else
                      model.invoke(messages(question["question"], docs)).content)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Empty model answer")
            row = {"id": qid, "question": question["question"], "answer": answer,
                   "reference": refs[qid], "seconds": time.perf_counter() - start,
                   "context": [{"text": d.page_content, **d.metadata} for d in docs or []]}
            rows.append(row)
            by_id[qid] = row
            write_json(path, rows)
            time.sleep(config["request_interval"])
        if "judge" not in row:
            row["judge"] = judge_answer(judge, row["question"], row["answer"], refs[qid])
            write_json(path, rows)
            time.sleep(config["request_interval"])
        print(f"{mode}: {len([r for r in rows if 'judge' in r])}/{len(questions)}", flush=True)
    metrics = lexical_metrics([r["answer"] for r in rows], [r["reference"] for r in rows])
    metrics.update(n=len(rows), judge_mean=mean(r["judge"]["score"] for r in rows),
                   mean_seconds=mean(r["seconds"] for r in rows), mode=mode)
    write_json(output / f"{mode}_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--mode", required=True, choices=["validate", "groq_zero", "local_zero", "groq_rag"])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.mode == "validate":
        articles, questions, _ = load_dataset(config)
        print(f"Valid dataset: {len(articles)} articles; {len(questions)} questions")
    else:
        run(config, args.mode)


if __name__ == "__main__":
    main()
