# news-return-pipeline

A lightweight pipeline for engineering-grade preprocessing of raw news headlines and market prices.
It creates a daily aggregated dataset with forward returns and supports strict temporal train/val/test splitting by year.

## Raw CSV schema requirements
Input CSV files must contain these columns:
- `date` (parseable date/time)
- `headline` (string)
- `close` (float)

Extra columns are allowed but ignored.

## Run preprocessing
From repo root:

```bash
pip install -e .
python -m news_return_pipeline.scripts.run_preprocess --input data/raw/sample.csv --output data/processed/daily_agg.csv --k_forward 5
```

## Output columns
The processed CSV contains:
- `date`
- `year`
- `text` (headlines for each day joined by ` [SEP] `)
- `n_headlines`
- `close` (last close observed for that date)
- `ret_k` (k-forward return, `close.shift(-k)/close - 1`)
