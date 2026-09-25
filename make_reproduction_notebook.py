"""Create a clean Colab entry point without overwriting the executed notebook."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md('# Повторение RAG-эксперимента\n\nВыберите GPU T4 и добавьте `GROQ_API_KEY` в Colab Secrets с доступом для ноутбука. '
       'Выполняйте ячейки сверху вниз. Данные уже в репозитории. Эта версия запускает новый эксперимент; '
       'исходный выполненный прогон 25.09.2026 сохранён в `01_colab.ipynb`. '
       'Новые ответы и оценки могут отличаться из-за внешнего API.'),
    code('''from pathlib import Path
import os, sys, json, subprocess
root = Path('/content/rag-homework-reproduce')
if not root.exists():
    subprocess.run(['git', 'clone', '--branch', 'rag-experiments',
                    'https://github.com/evelinashakhnazaryan/rag-homework.git', str(root)], check=True)
os.chdir(root)
print('Project:', root.name)
subprocess.run(['git', 'rev-parse', 'HEAD'], check=True)
subprocess.run(['nvidia-smi'], check=True)'''),
    code('''installation = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-colab.txt'],
                              capture_output=True, text=True)
print(installation.stdout[-4000:] + installation.stderr[-4000:])
assert installation.returncode == 0'''),
    code('''from google.colab import userdata
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
config = json.loads(Path('configs/default.json').read_text(encoding='utf-8'))
config['results_dir'] = 'results_reproduction'
config_path = Path('configs/reproduction.json')
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
def run_stage(mode):
    command = [sys.executable, '-u', '-m', 'src.run', '--config', str(config_path), '--mode', mode]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    assert status == 0, f'{mode}: ошибка, повторите ячейку для продолжения с checkpoint'
run_stage('validate')
print(json.dumps(config, ensure_ascii=False, indent=2))'''),
    md('## API без RAG'), code("run_stage('groq_zero')"),
    md('## Локальная Qwen через vLLM'), code("run_stage('local_zero')"),
    md('## Та же API-модель с LangChain RAG'), code("run_stage('groq_rag')"),
    md('## Итоговый отчёт'),
    code('''result = subprocess.run([sys.executable, '-m', 'src.report', '--config', str(config_path)],
                        capture_output=True, text=True)
assert result.returncode == 0, result.stderr
from IPython.display import Markdown, display
display(Markdown(Path(config['results_dir'], 'REPORT.md').read_text(encoding='utf-8')))
import importlib.metadata, platform
env = {'python': platform.python_version(),
       'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
Path(config['results_dir'], 'environment_final.json').write_text(json.dumps(env, indent=2), encoding='utf-8')'''),
    md('## Сохранение\n\nСкачайте `.ipynb` через меню Файл → Скачать: выводы сохранятся вместе с кодом. '
       'Следующая ячейка скачивает все результаты. Для нового независимого прогона смените `results_dir`; '
       'повторный запуск с тем же каталогом продолжает уже сохранённые ответы.'),
    code('''import zipfile
from google.colab import files
with zipfile.ZipFile('/content/rag-reproduction-results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for p in Path(config['results_dir']).glob('*'):
        if p.is_file():
            archive.write(p, str(p))
files.download('/content/rag-reproduction-results.zip')'''),
]
nb.metadata.update(kernelspec={'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   language_info={'name': 'python'}, accelerator='GPU')
nbf.write(nb, Path('notebooks/02_reproduce.ipynb'))
print('Created clean reproduction notebook; executed notebook preserved.')
