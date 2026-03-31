import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# HYPERPARAMETERS
SEQ_LEN = 10
EPOCHS = 50
LEARNING_RATE = 0.01
TRAIN_SPLIT = 0.85

# LOAD DATA
data_file = os.path.join("data", "processed", "model_dataset.csv")
df = pd.read_csv(data_file)

# Correct column names for your CSV
prices = df['close'].values.reshape(-1,1)
sentiments = df['sentiment_mean'].values.reshape(-1,1)  # FinBERT feature

# NORMALIZE
scaler_prices = MinMaxScaler()
prices_scaled = scaler_prices.fit_transform(prices)

scaler_sent = MinMaxScaler()
sentiments_scaled = scaler_sent.fit_transform(sentiments)

# TRAIN/TEST SPLIT
split_idx = int(len(prices_scaled) * TRAIN_SPLIT)
train_prices = prices_scaled[:split_idx]
test_prices = prices_scaled[split_idx:]
train_sent = sentiments_scaled[:split_idx]
test_sent = sentiments_scaled[split_idx:]

# CREATE SEQUENCES
def create_sequences(prices, seq_len, sentiments=None):
    X, y = [], []
    for i in range(len(prices) - seq_len):
        seq = prices[i:i+seq_len].copy()
        if sentiments is not None:
            seq = np.hstack((seq, sentiments[i:i+seq_len]))
        X.append(seq)
        y.append(prices[i+seq_len])
    return np.array(X), np.array(y)

# Prepare sequences for each model
X_train_mlp, y_train_mlp = create_sequences(train_prices, SEQ_LEN)
X_test_mlp, y_test_mlp = create_sequences(test_prices, SEQ_LEN)
X_train_mlp = X_train_mlp.reshape(X_train_mlp.shape[0], -1)
X_test_mlp = X_test_mlp.reshape(X_test_mlp.shape[0], -1)

X_train_lstm, y_train_lstm = create_sequences(train_prices, SEQ_LEN)
X_test_lstm, y_test_lstm = create_sequences(test_prices, SEQ_LEN)

X_train_bert, y_train_bert = create_sequences(train_prices, SEQ_LEN, train_sent)
X_test_bert, y_test_bert = create_sequences(test_prices, SEQ_LEN, test_sent)

# MODEL CREATION FUNCTIONS
def create_mlp(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    return model

def create_lstm(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.LSTM(50, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(30, activation='tanh', return_sequences=False),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    return model

# TRAIN, PREDICT & EVALUATE
def train_eval(model_func, X_train, y_train, X_test, y_test, is_lstm=False):
    if is_lstm:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
        input_shape = (X_train.shape[1], X_train.shape[2])
    else:
        input_shape = X_train.shape[1:]
    model = model_func(input_shape)
    model.fit(X_train, y_train, epochs=EPOCHS, verbose=0)
    preds = model.predict(X_test)
    preds_rescaled = scaler_prices.inverse_transform(preds)
    y_test_rescaled = scaler_prices.inverse_transform(y_test)
    mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
    mape = mean_absolute_percentage_error(y_test_rescaled, preds_rescaled)
    acc = 1 - mape
    return mae, mape, acc, preds_rescaled

# Run models
mlp_metrics = train_eval(create_mlp, X_train_mlp, y_train_mlp, X_test_mlp, y_test_mlp)
lstm_metrics = train_eval(create_lstm, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, is_lstm=True)
bert_metrics = train_eval(create_lstm, X_train_bert, y_train_bert, X_test_bert, y_test_bert, is_lstm=True)

# RESULTS TABLE
results = pd.DataFrame({
    "Model": ["MLP", "LSTM", "FinBERT-LSTM"],
    "MAE": [mlp_metrics[0], lstm_metrics[0], bert_metrics[0]],
    "MAPE": [mlp_metrics[1], lstm_metrics[1], bert_metrics[1]],
    "Accuracy": [mlp_metrics[2], lstm_metrics[2], bert_metrics[2]]
})
print(results)

# PLOT ALL THREE
plt.figure(figsize=(12,6))
plt.plot(scaler_prices.inverse_transform(y_test_mlp), color='black', linewidth=2.5, label='Actual')
plt.plot(mlp_metrics[3], color='green', linewidth=2, label='MLP')
plt.plot(lstm_metrics[3], color='orange', linewidth=2, label='LSTM')
plt.plot(bert_metrics[3], color='red', linewidth=2, label='FinBERT-LSTM')
plt.xlabel("Timestep")
plt.ylabel("Price")
plt.title("S&P 500 Prediction Comparison")
plt.legend()
plt.tight_layout()


output_dir = os.path.join("data", "processed")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "model_comparison.png")
plt.savefig(output_file)

plt.show()
