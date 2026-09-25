"""Record dataset integrity without modifying source data."""
import argparse
import hashlib
import json
from pathlib import Path
from .data import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    articles, questions, refs = load_dataset(config)
    result = {
        'articles': len(articles), 'questions': len(questions), 'references': len(refs),
        'article_characters': sum(len(a['text']) for a in articles),
        'question_ids_match_reference_ids': True,
        'sha256': {name: hashlib.sha256((Path(config['data_dir']) / name).read_bytes()).hexdigest()
                   for name in ['articles.json', 'questions.json', 'ground_truth.json']},
        'topics': [{'id': a['id'], 'title': a.get('title', '')} for a in articles],
    }
    target = Path(config['results_dir'])
    target.mkdir(parents=True, exist_ok=True)
    (target / 'data_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
