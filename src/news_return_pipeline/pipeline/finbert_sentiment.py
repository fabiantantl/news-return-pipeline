# Calling FinBERT

from transformers import pipeline


def load_finbert():
    """Load FinBERT sentiment model."""
    return pipeline("text-classification", model="ProsusAI/finbert")


def compute_finbert_sentiment(df, text_column="title", batch_size=64):
    """Compute FinBERT sentiment for each row."""
    clf = load_finbert()

    results = clf(df[text_column].tolist(), batch_size=batch_size)

    df = df.copy()
    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]

    return df