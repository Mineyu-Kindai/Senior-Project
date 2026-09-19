了解しました。今後は、あなたの卒業研究プロジェクトのルートディレクトリを次として扱います。

```text
/home/my/senior-project
```

プロンプトに表示されている `(senior-project)` は、現在そのプロジェクトの仮想環境が有効になっていることを示している可能性があります。

## 推奨ディレクトリ構成

まずは、プロジェクト直下に用途別のディレクトリを作るのがおすすめです。

```text
/home/my/senior-project/
├── pyproject.toml
├── uv.lock
├── README.md
├── src/
│   └── senior_project/
│       ├── __init__.py
│       ├── detection.py
│       ├── attention.py
│       └── models.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_experiment.py
├── notebooks/
│   └── attention_analysis.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── configs/
│   └── experiment.yaml
├── results/
│   ├── figures/
│   ├── metrics/
│   └── logs/
└── tests/
    └── test_detection.py
```

Pythonの再利用可能なコードを`src/`配下に置き、テストを`tests/`に分ける構成は、Pythonプロジェクトで一般的な構成です。 [packaging.python](https://packaging.python.org/en/latest/tutorials/packaging-projects/?highlight=pyproject.toml)

## 何をどこに作るか

### Pythonの処理本体

長く使い回すコードは、次に作ります。

```text
/home/my/senior-project/src/senior_project/
```

例えば、Attention抽出処理なら、

```text
src/senior_project/attention.py
```

プロンプトインジェクションの分類処理なら、

```text
src/senior_project/detection.py
```

のようにします。

### 実験を実行するファイル

実験を開始するためのスクリプトは、次に作ると整理しやすいです。

```text
/home/my/senior-project/scripts/
```

例えば、

```text
scripts/train_classifier.py
scripts/evaluate_classifier.py
scripts/extract_attention.py
scripts/run_experiment.py
```

のようにします。

ただし、最初の小さな実験では、プロジェクト直下に`experiment.py`を置いても問題ありません。実験数が増えてきたら、`scripts/`へ整理すれば大丈夫です。

### Jupyter Notebook

Jupyter Notebookは次に作ります。

```text
/home/my/senior-project/notebooks/
```

例えば、

```text
notebooks/01_dataset_check.ipynb
notebooks/02_attention_visualization.ipynb
notebooks/03_result_analysis.ipynb
```

Notebook内に大量の処理を書きすぎず、主要な処理は`src/senior_project/`のPythonファイルに分けると、再利用しやすくなります。

### データセット

元データと加工済みデータは分けて保存します。

```text
/home/my/senior-project/data/raw/
```

ここには、ダウンロードした元のデータや加工前のデータを置きます。

```text
data/raw/prompts.jsonl
data/raw/injection_dataset.csv
```

加工後のデータは次に置きます。

```text
/home/my/senior-project/data/processed/
```

例えば、

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
```

元データを上書きしないことが重要です。再実験やデータ処理の検証がしやすくなります。

### 実験結果

モデルの出力、評価結果、グラフ、ログは次に置きます。

```text
/home/my/senior-project/results/
```

例えば、

```text
results/metrics/exp001_metrics.json
results/figures/confusion_matrix.png
results/logs/exp001.log
```

実験用のコードと結果を分けることで、どのコードからどの結果が生成されたかを追跡しやすくなります。

## まず作成するコマンド

現在のディレクトリが`/home/my/senior-project`であることを確認します。

```bash
pwd
```

次のように表示されれば正しい場所です。

```text
/home/my/senior-project
```

ディレクトリを作成します。

```bash
mkdir -p src/senior_project
mkdir -p scripts
mkdir -p notebooks
mkdir -p data/raw
mkdir -p data/processed
mkdir -p configs
mkdir -p results/figures
mkdir -p results/metrics
mkdir -p results/logs
mkdir -p tests
```

Pythonパッケージとして認識させるため、`__init__.py`を作ります。

```bash
touch src/senior_project/__init__.py
```

現在の構成を確認します。

```bash
tree -a -L 3
```

`tree`がない場合は、先にインストールします。

```bash
sudo apt update
sudo apt install -y tree
```

## 最初の実験ファイル

例えば、最初の確認用実験ファイルは次の場所に作れます。

```text
/home/my/senior-project/scripts/experiment_ hello.py
```

ただし、ファイル名に空白は使わない方がよいので、次の名前にします。

```text
/home/my/senior-project/scripts/experiment_hello.py
```

作成するには、

```bash
nano scripts/experiment_hello.py
```

またはVS Codeから、

```bash
code scripts/experiment_hello.py
```

内容は次のようにします。

```python
import sys

print(f"Python: {sys.version}")
print("senior-project experiment is running.")
```

実行します。

```bash
uv run python scripts/experiment_hello.py
```

## 現時点でのおすすめ

今の段階では、次のように使い分けるのがよいです。

- **再利用するPython処理**：`src/senior_project/`
- **実験を実行するスクリプト**：`scripts/`
- **対話的な分析**：`notebooks/`
- **元データ**：`data/raw/`
- **加工済みデータ**：`data/processed/`
- **実験結果**：`results/`
- **テストコード**：`tests/`

最初の実験ファイルは、まずここに作るのがおすすめです。

```text
/home/my/senior-project/scripts/
```

一方、Attention抽出や分類器など、複数の実験から呼び出す処理は、こちらに作ります。

```text
/home/my/senior-project/src/senior_project/
```

[Python Packaging User Guide: Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/) [packaging.python](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)