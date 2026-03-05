# news-return-pipeline

A lightweight pipeline for engineering-grade preprocessing of raw news headlines and market prices.
It creates a daily aggregated dataset with forward returns and supports strict temporal train/val/test splitting by year.

## Raw CSV schema requirements
Input CSV files must contain these columns:
- `date` (parseable date/time)
- `headline` (string)
- `close` (float)

Extra columns are allowed but ignored.

## Import Kaggle prototype dataset
Use this one-time importer to download the FindKG GDELT-based financial news dataset and place it in `data/raw/`:

```bash
python -m pip install kagglehub
python -m news_return_pipeline.scripts.import_prototype_kaggle
# optional override:
# python -m news_return_pipeline.scripts.import_prototype_kaggle --dataset <owner/slug> --output-filename custom.csv
```

This writes `data/raw/kaggle_findkg_news_clean.csv` with normalized columns (`date`, `headline`, `close`) so it can be used directly by the preprocessing pipeline.

## Run preprocessing
From repo root:

```bash
python -m pip install -e .
python -m news_return_pipeline.scripts.run_preprocess
```

Notes:
- The raw input file must exist in `data/raw/`.
- The default raw filename is defined in `Config` as `raw_filename` (default: `kaggle_findkg_news_clean.csv`).
- Processed output is written to `data/processed/`.
- The default processed filename is defined in `Config` as `processed_filename` (default: `daily_agg.csv`).

## Output columns
The processed CSV contains:
- `date`
- `year`
- `text` (headlines for each day joined by ` [SEP] `)
- `n_headlines`
- `close` (last close observed for that date)
- `ret_k` (k-forward return, `close.shift(-k)/close - 1`)


## Windows (without editable install)
If you prefer not to install the package in editable mode, run commands with `PYTHONPATH=src` style module resolution:

```powershell
$env:PYTHONPATH="src"
python -m news_return_pipeline.scripts.import_prototype_kaggle
python -m news_return_pipeline.scripts.run_preprocess
```
