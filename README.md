# news-return-pipeline

A lightweight pipeline for engineering-grade preprocessing of financial news articles and market prices.

The pipeline downloads the FindKG GDELT-based financial news dataset from Kaggle, loads article data stored in Parquet format, and prepares a dataset suitable for downstream machine learning experiments.

It is designed for research on whether financial news signals can predict market returns.

## Kaggle authentication

This project downloads data using **kagglehub**.

Create a `.env` file in the repository root:

KAGGLE_API_TOKEN=your_kaggle_token_here

This token allows kagglehub to authenticate and download datasets.

## Dataset

The pipeline downloads the Kaggle dataset:

verracodeguacas/findkg-gdelt-based-financial-news-and-mentions

The dataset contains multiple files including:

- articles_YYYY.parquet (main article dataset)
- article_org_edges_YYYY.parquet (entity relationship edges)
- findkg_calendar_status.csv (dataset status metadata)

The pipeline reads **articles_*.parquet** files containing financial news articles extracted from GDELT.

## Raw dataset schema

The article parquet files contain columns such as:

- article_id
- seendate
- seendate_dt
- date
- weekday
- title
- domain
- url
- url_host
- url_path
- bucket
- sourcecountry
- lang
- gkg_orgs
- gkg_themes
- gkg_persons
- gkg_dates
- has_gkg_match
- n_gkg_orgs

For modelling purposes the pipeline maps the raw schema into a simplified format.

Key mappings include:

- `date` → article publication date
- `title` → news headline text
- additional metadata fields may be used for feature engineering

## Build dataset (single pipeline entrypoint)

From repo root:

```bash
python -m pip install -e .
python -m news_return_pipeline.scripts.build_dataset
```

This command performs the full pipeline:

1. Downloads the Kaggle dataset.
2. Loads article parquet files.
3. Normalizes the schema.
4. Aggregates headlines by date.
5. Writes output to `data/processed/dataset.csv`.

## Output columns

The processed CSV contains:

- date
- year
- text (headlines for each day joined by `[SEP]`)
- n_headlines
- additional engineered features depending on preprocessing steps

## Authors

Fabian Tan  
Hector Chen


Notes for Authors Use 
for now: use neutral→0
preferred later: use p_pos - p_neg

map weekend headings to the next day