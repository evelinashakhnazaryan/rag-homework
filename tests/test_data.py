import json
import tempfile
import unittest
from pathlib import Path
from src.data import records, load_dataset, tokens


class DataTests(unittest.TestCase):
    def test_preserves_russian_and_numbers(self):
        self.assertEqual(tokens("Отпуск: 28 дней, 2026 год"), ["отпуск", "28", "дней", "2026", "год"])

    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.json"
            path.write_text(json.dumps([{"id": 1, "question": "A"}, {"id": 1, "question": "B"}]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                records(path, "question")

    def test_reference_join_and_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in {
                "articles": {"a": "Article"},
                "questions": {"q1": "First?", "q2": "Second?"},
                "ground_truth": {"q2": "Second!", "q1": "First!"},
            }.items():
                (root / f"{name}.json").write_text(json.dumps(value))
            config = {"data_dir": directory, "expected_articles": 1, "expected_questions": 2}
            _, _, refs = load_dataset(config)
            self.assertEqual(refs["q1"], "First!")
            (root / "ground_truth.json").write_text('{"wrong":"answer"}')
            with self.assertRaisesRegex(ValueError, "IDs"):
                load_dataset(config)


if __name__ == "__main__":
    unittest.main()
