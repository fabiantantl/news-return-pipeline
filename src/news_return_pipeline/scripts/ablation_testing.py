import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
SEQ_LEN = 10
EPOCHS = 50
LEARNING_RATE = 0.01
TRAIN_SPLIT = 0.85

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
data_file = os.path.join(BASE_DIR, "data", "processed", "model_dataset.csv")

df = pd.read_csv(data_file)

prices = df["close"].values.reshape(-1, 1)
sentiments = df["sentiment_mean"].values.reshape(-1, 1)

# =========================
# NORMALIZATION
# =========================
scaler_prices = MinMaxScaler()
prices_scaled = scaler_prices.fit_transform(prices)

scaler_sent = MinMaxScaler()
sentiments_scaled = scaler_sent.fit_transform(sentiments)

# =========================
# SPLIT
# =========================
split_idx = int(len(prices_scaled) * TRAIN_SPLIT)

train_prices = prices_scaled[:split_idx]
test_prices = prices_scaled[split_idx:]

train_sent = sentiments_scaled[:split_idx]
test_sent = sentiments_scaled[split_idx:]

# =========================
# SEQUENCE BUILDERS
# =========================
def create_sequences(prices, seq_len, sentiments=None):
    X, y = [], []
    for i in range(len(prices) - seq_len):
        seq = prices[i:i+seq_len].copy()
        if sentiments is not None:
            seq = np.hstack((seq, sentiments[i:i+seq_len]))
        X.append(seq)
        y.append(prices[i+seq_len])
    return np.array(X), np.array(y)

def create_sent_only(prices, sentiments, seq_len):
    X, y = [], []
    for i in range(len(prices) - seq_len):
        X.append(sentiments[i:i+seq_len])
        y.append(prices[i+seq_len])
    return np.array(X), np.array(y)

def shuffle_sent(sent):
    s = sent.copy()
    np.random.shuffle(s)
    return s

# =========================
# DATASETS
# =========================

# MLP (price only)
X_train_mlp, y_train_mlp = create_sequences(train_prices, SEQ_LEN)
X_test_mlp, y_test_mlp = create_sequences(test_prices, SEQ_LEN)

X_train_mlp = X_train_mlp.reshape(X_train_mlp.shape[0], -1)
X_test_mlp = X_test_mlp.reshape(X_test_mlp.shape[0], -1)

# LSTM (price only)
X_train_lstm, y_train_lstm = create_sequences(train_prices, SEQ_LEN)
X_test_lstm, y_test_lstm = create_sequences(test_prices, SEQ_LEN)

# FinBERT-LSTM
X_train_bert, y_train_bert = create_sequences(train_prices, SEQ_LEN, train_sent)
X_test_bert, y_test_bert = create_sequences(test_prices, SEQ_LEN, test_sent)

# Shuffled sentiment
train_sent_shuf = shuffle_sent(train_sent)
test_sent_shuf = shuffle_sent(test_sent)

X_train_shuf, y_train_shuf = create_sequences(train_prices, SEQ_LEN, train_sent_shuf)
X_test_shuf, y_test_shuf = create_sequences(test_prices, SEQ_LEN, test_sent_shuf)

# Sentiment only
X_train_sent, y_train_sent = create_sent_only(train_prices, train_sent, SEQ_LEN)
X_test_sent, y_test_sent = create_sent_only(test_prices, test_sent, SEQ_LEN)

# MLP + sentiment
X_train_mlp_sent, y_train_mlp_sent = create_sequences(train_prices, SEQ_LEN, train_sent)
X_test_mlp_sent, y_test_mlp_sent = create_sequences(test_prices, SEQ_LEN, test_sent)

X_train_mlp_sent = X_train_mlp_sent.reshape(X_train_mlp_sent.shape[0], -1)
X_test_mlp_sent = X_test_mlp_sent.reshape(X_test_mlp_sent.shape[0], -1)

# =========================
# MODELS
# =========================
def create_mlp(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE)
    )
    return model

def create_lstm(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(30),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE)
    )
    return model

# =========================
# TRAIN / EVAL
# =========================
def train_eval(model_fn, X_train, y_train, X_test, y_test, is_lstm=False):

    if is_lstm:
        input_shape = (X_train.shape[1], X_train.shape[2])
    else:
        input_shape = X_train.shape[1:]

    model = model_fn(input_shape)
    model.fit(X_train, y_train, epochs=EPOCHS, verbose=0)

    preds = model.predict(X_test)

    preds = scaler_prices.inverse_transform(preds)
    y_true = scaler_prices.inverse_transform(y_test)

    mae = mean_absolute_error(y_true, preds)
    mape = mean_absolute_percentage_error(y_true, preds)
    acc = 1 - mape

    return mae, mape, acc, preds

# =========================
# RUN EXPERIMENTS
# =========================
mlp = train_eval(create_mlp, X_train_mlp, y_train_mlp, X_test_mlp, y_test_mlp)
lstm = train_eval(create_lstm, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, True)
bert = train_eval(create_lstm, X_train_bert, y_train_bert, X_test_bert, y_test_bert, True)
shuf = train_eval(create_lstm, X_train_shuf, y_train_shuf, X_test_shuf, y_test_shuf, True)
sent = train_eval(create_lstm, X_train_sent, y_train_sent, X_test_sent, y_test_sent, True)
mlp_sent = train_eval(create_mlp, X_train_mlp_sent, y_train_mlp_sent, X_test_mlp_sent, y_test_mlp_sent)

# =========================
# RESULTS TABLE
# =========================
results = pd.DataFrame({
    "Model": [
        "MLP",
        "LSTM",
        "FinBERT-LSTM",
        "Shuffled",
        "Sentiment Only",
        "MLP + Sentiment"
    ],
    "MAE": [mlp[0], lstm[0], bert[0], shuf[0], sent[0], mlp_sent[0]],
    "MAPE": [mlp[1], lstm[1], bert[1], shuf[1], sent[1], mlp_sent[1]],
    "Accuracy": [mlp[2], lstm[2], bert[2], shuf[2], sent[2], mlp_sent[2]]
})

print("\n===== MODEL COMPARISON =====")
print(results.to_string(index=False))

print("\nBEST MODEL:")
print(results.loc[results["MAPE"].idxmin()])

# =========================
# OUTPUT DIR
# =========================
out_dir = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(out_dir, exist_ok=True)

# =========================
# 🔥 CLEAN BAR CHART (NORMALIZED)
# =========================
scaler = MinMaxScaler()
norm = results.copy()
norm[["MAE", "MAPE", "Accuracy"]] = scaler.fit_transform(norm[["MAE", "MAPE", "Accuracy"]])

plt.figure(figsize=(11,6))

x = np.arange(len(norm))
w = 0.25

plt.bar(x - w, norm["MAE"], w, label="MAE (norm)")
plt.bar(x, norm["MAPE"], w, label="MAPE (norm)")
plt.bar(x + w, norm["Accuracy"], w, label="Accuracy (norm)")

plt.xticks(x, norm["Model"], rotation=25, ha="right")
plt.title("Ablation Study — Normalized Metrics")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "ablation_bar_clean.png"), dpi=300)
plt.close()

# =========================
# 🔥 CLEAN PREDICTIONS PLOT
# =========================
plt.figure(figsize=(13,6))

actual = scaler_prices.inverse_transform(y_test_mlp)

plt.plot(actual, label="Actual", linewidth=3, color="black")

plt.plot(mlp[3], label="MLP", alpha=0.8)
plt.plot(lstm[3], label="LSTM", alpha=0.9)
plt.plot(bert[3], label="FinBERT-LSTM", alpha=0.85)
plt.plot(shuf[3], label="Shuffled", alpha=0.7)
plt.plot(sent[3], label="Sentiment Only", alpha=0.7)
plt.plot(mlp_sent[3], label="MLP + Sentiment", alpha=0.85)

plt.title("Ablation Prediction Comparison")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(alpha=0.2)
plt.legend(ncol=2, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ablation_predictions_clean.png"), dpi=300)
plt.close()

# =========================
# SAVE CSV
# =========================
results.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)

print("\nSaved outputs to:", out_dir)
