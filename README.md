# news-return-pipeline

A lightweight pipeline for engineering-grade preprocessing of raw news headlines and market prices.
It creates a daily aggregated dataset with forward returns and supports strict temporal train/val/test splitting by year.

## Raw CSV schema requirements
Input CSV files must contain these columns:
- `date` (parseable date/time)
- `headline` (string)
- `close` (float)

Extra columns are allowed but ignored.

## Build dataset (single pipeline entrypoint)
From repo root:

```bash
python -m pip install -e .
python -m news_return_pipeline.scripts.build_dataset
```

This command performs the full pipeline:
1. Downloads the prototype Kaggle dataset.
2. Normalizes schema to `date`, `headline`, `close`.
3. Preprocesses into daily aggregates + forward returns.
4. Writes output to `data/processed/dataset.csv`.

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
python -m news_return_pipeline.scripts.build_dataset
```
