"""Build an evidence-based report only after all three runs complete."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["results_dir"])
    modes = ["groq_zero", "local_zero", "groq_rag"]
    metrics = {m: json.loads((root / f"{m}_metrics.json").read_text(encoding="utf-8")) for m in modes}
    if any(v["n"] != config["expected_questions"] for v in metrics.values()):
        raise ValueError("Incomplete evaluation")
    lines = ["# Результаты эксперимента", "", "| Режим | N | BLEU /100 | ROUGE-1 | ROUGE-2 | ROUGE-L | Judge /4 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, row in metrics.items():
        lines.append(f"| {mode} | {row['n']} | {row['bleu']:.2f} | {row['rouge1']:.3f} | {row['rouge2']:.3f} | {row['rougeL']:.3f} | {row['judge_mean']:.2f} |")
    delta = metrics["groq_rag"]["judge_mean"] - metrics["groq_zero"]["judge_mean"]
    lines += ["", f"При добавлении RAG средняя оценка судьи изменилась на {delta:+.2f} балла из 4.",
              "Сравнение использует одну и ту же API-модель и одинаковые вопросы. Это наблюдение на данном наборе, не доказательство статистической значимости.",
              "", "## Примеры ошибок RAG"]
    rows = json.loads((root / "groq_rag.json").read_text(encoding="utf-8"))
    errors = [row for row in rows if row["judge"]["score"] < 4]
    for row in sorted(errors, key=lambda r: r["judge"]["score"])[:5]:
        lines += ["", f"### Вопрос {row['id']}", row["question"], "", f"Ответ: {row['answer']}",
                  "", f"Эталон: {row['reference']}", "", f"Судья: {row['judge']['reason']}"]
    lines += ["", "## Ограничения и улучшения", "",
              "BLEU и ROUGE измеряют совпадения слов и не учитывают все допустимые перефразировки. "
              "LLM-судья может ошибаться; нужна ручная проверка ошибок и второй независимый судья. "
              "Далее: отдельный validation set для выбора chunk_size/top_k, гибридный BM25+dense поиск, reranker, "
              "разметка релевантных документов для Recall@k и проверка обоснованности ответа источниками."]
    text = "\n".join(lines) + "\n"
    (root / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
