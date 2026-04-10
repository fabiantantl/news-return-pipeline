# news-return-pipeline

A reproducible pipeline for studying whether financial news headline sentiment helps predict short-horizon market returns.

This repo builds a leakage-aware daily dataset from financial headlines and S&P 500 prices using FinBERT sentiment, trading-day alignment, and temporal splits.

## What this repo does

- downloads and preprocesses financial news and price data
- scores headlines with **FinBERT**
- aligns non-trading-day news to the next trading day
- aggregates headline sentiment into daily features
- builds a model-ready dataset with **5-day forward returns**
- trains simple baseline models
- includes exploratory sequence-model and ablation scripts

## Main setup

The main pipeline is built around:

- **Headlines only**
- **Frozen `ProsusAI/finbert`**
- **S&P 500 index prediction**
- **5-day forward close-to-close return**
- **Strict temporal handling**
- **Coverage from 2020 to 2026**

## Repo structure

```text
src/news_return_pipeline/
├── datasets/
├── pipeline/
├── models/
├── evaluation/
└── scripts/
    ├── build_dataset.py
    ├── train_baseline.py
    ├── train.py
    └── ablation_testing.py
```

## Installation

```bash
python -m pip install -e .
```

## Kaggle setup

Create a `.env` file in the repo root:

```bash
KAGGLE_API_TOKEN=your_kaggle_token_here
```

## Data sources

- **Financial news:** `verracodeguacas/findkg-gdelt-based-financial-news-and-mentions`
- **S&P 500 prices:** `andrewmvd/sp-500-stocks`
- **Multi-stock prices:** `nelgiriyewithana/world-stock-prices-daily-updating`

## Build the dataset

Run the full data pipeline with:

```bash
python -m news_return_pipeline.scripts.build_dataset
```

This pipeline:

- downloads yearly news files from 2020 to 2026
- preprocesses headlines and prices
- runs FinBERT sentiment scoring
- aligns news to trading days
- aggregates daily news features
- creates `target_return_5d`
- saves processed datasets

## Train baselines

```bash
python -m news_return_pipeline.scripts.train_baseline
```

Current baselines include:

- mean baseline
- lag-1 linear regression baseline

## Exploratory models

The repo also includes exploratory TensorFlow scripts:

```bash
python -m news_return_pipeline.scripts.train
python -m news_return_pipeline.scripts.ablation_testing
```

These are separate from the main dataset pipeline and are mainly for sequence modelling and ablation analysis.

## Main outputs

Typical outputs include:

```bash
data/processed/model_dataset.csv
data/processed/baseline_metrics.json
data/processed/baseline_predictions_val.csv
data/processed/baseline_predictions_test.csv
```

Other intermediate artifacts may include:

```bash
data/processed/news_preprocessed.csv
data/processed/news_finbert_scored.csv
data/processed/news_aligned.csv
data/processed/news_daily_features.csv
data/processed/sp500_preprocessed.csv
data/processed/sp500_with_targets.csv
```

## Final dataset

The final daily dataset is built around S&P 500 trading dates and may include fields such as:

- `date`
- `open`
- `close`
- `target_return_5d`
- `sentiment_mean`
- `sentiment_std`
- `n_headlines`
- `frac_positive`
- `frac_negative`
- `frac_neutral`
- `text`

## Notes

This repo currently contains two experimental tracks:

1. **Main pipeline**
   - builds the leakage-aware daily dataset
   - uses FinBERT features, trading-day alignment, and saved intermediate artifacts

2. **Exploratory sequence scripts**
   - `train.py` and `ablation_testing.py`
   - use chronological splits and additional ablations

## Current takeaway

So far, the strongest signal appears to come from **price history**, while sentiment features have not shown consistent gains under the current setup. The repo is structured to make that comparison explicit through temporal controls and ablation testing.

## Quick start

```bash
python -m pip install -e .
python -m news_return_pipeline.scripts.build_dataset
python -m news_return_pipeline.scripts.train_baseline
```

## Authors

Fabian Tan  
Hector Chen