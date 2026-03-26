# Calling FinBERT

from transformers import pipeline


def load_finbert():
    """Load FinBERT sentiment model."""
    return pipeline("text-classification", model="ProsusAI/finbert")


def compute_finbert_sentiment(df, text_column="title", batch_size=32):
    clf = load_finbert()

    texts = df[text_column].tolist()
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_result = clf(batch)
        results.extend(batch_result)

        print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)}")

    df = df.copy()
    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]

    return df