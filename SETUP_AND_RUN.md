# Setup and Definitive Run

## 1. Raw files

Keep the three original CSV files unchanged in `data/raw/`:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

The pipeline validates and hashes them. Do not edit or resave them in Excel.

## 2. Environment

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Tests

```bat
python -m pytest tests -q
```

## 4. Commit the methodology before the definitive run

```bat
git init
git add .gitignore src config tests tools requirements.txt run.bat Makefile pytest.ini RESEARCH_GRADE_V2_UPGRADE.md CODE_MAP.md CORRECTION_GUIDE.md SETUP_AND_RUN.md
git commit -m "Research-grade V2 methodology"
```

## 5. Run once

```bat
run.bat
```

The pipeline archives previous final outputs before generating a clean run.

## 6. Verify

Open `results/final/run_completion.json` and confirm `status` is `success`.
