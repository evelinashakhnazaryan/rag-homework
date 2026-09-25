import unittest
from src.evaluation import lexical_metrics, judge_answer


class EvaluationTests(unittest.TestCase):
    def test_cyrillic_exact_match(self):
        metrics = lexical_metrics(["Отпуск составляет 28 дней"], ["Отпуск составляет 28 дней"])
        self.assertAlmostEqual(metrics['rougeL'], 1)
        self.assertAlmostEqual(metrics['bleu'], 100)

    def test_changed_number_is_not_exact_match(self):
        metrics = lexical_metrics(["Отпуск составляет 14 дней"], ["Отпуск составляет 28 дней"])
        self.assertLess(metrics['rougeL'], 1)

    def test_invalid_judge_score_is_rejected(self):
        class FakeJudge:
            def invoke(self, messages):
                class Response:
                    content = '{"score":99,"reason":"invalid"}'
                return Response()
        with self.assertRaisesRegex(ValueError, 'three times'):
            judge_answer(FakeJudge(), 'Question', 'Answer', 'Reference')


if __name__ == '__main__':
    unittest.main()
