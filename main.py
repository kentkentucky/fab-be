from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow import keras

import numpy as np
import pandas as pd
import joblib

from core.fetch_data import fetch_market_data, fetch_stock_news
from core.feature_engineering import merge_data
from core.risk_scoring import calculate_risk, Profile
from validation.firebase_admin import verify_token

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# function to create lstm sequence
def lstm_sequence(data, labels, length=60):
    # initialise x and y seq
    X_seq, y_seq = [], []

    # ensure consistent indexing
    labels_array = labels.values if hasattr(labels, 'values') else np.array(labels)

    # create sequence
    for i in range(length, len(data)):
        # append sequence into x seq
        X_seq.append(data[i - length: i])
        # append sequence into y seq
        y_seq.append(labels_array[i])

    # return sequence array
    return np.array(X_seq), np.array(y_seq)

# generate predictions using the loaded models
def generate_strategies(rf_models, lstm_models, backtest_data, rf_threshold, lstm_threshold):
    strategies = {}

    for ticker in rf_models.keys():
        if ticker not in lstm_models or ticker not in backtest_data:
            continue

        print(f'Generating predictions for {ticker}...')

        try:
            data = backtest_data[ticker].copy()

            # rf feature columns
            rf_feature_columns = [
                'rsi', 'sma_50', 'adx', 'volume', 'corr', 'prev_open_close', 'prev_close_high',
                'prev_close_low', 'momentum', 'volatility', 'sentiment_mean', 'sentiment_std',
                'news_count', 'sentiment_strength', 'sentiment_volume'
            ]

            # lstm feature columns
            lstm_feature_columns = [
                'rsi', 'sma_50', 'adx', 'volume', 'corr', 'momentum', 
                'volatility', 'macd', 'returns', 'sentiment_mean', 'sentiment_std',
                'news_count', 'sentiment_strength', 'sentiment_volume'
            ]

            # Check if all required features exist
            missing_rf = [col for col in rf_feature_columns if col not in data.columns]
            missing_lstm = [col for col in lstm_feature_columns if col not in data.columns]
            
            if missing_rf:
                print(f"Missing RF features for {ticker}: {missing_rf}")
                continue
            if missing_lstm:
                print(f"Missing LSTM features for {ticker}: {missing_lstm}")
                continue

            # prepare rf features
            X_rf = data[rf_feature_columns].dropna()
            X_rf_scaled = rf_models[ticker]['scaler'].transform(X_rf)

            # get rf predictions
            rf_proba = rf_models[ticker]['model'].predict_proba(X_rf_scaled)[:, 1]

            # prepare lstm features
            X_lstm = data[lstm_feature_columns].dropna()
            X_lstm_scaled = lstm_models[ticker]['feature_scaler'].transform(X_lstm)

            # create lstm sequence
            X_lstm_seq, _ = lstm_sequence(X_lstm_scaled, np.zeros(len(X_lstm_scaled)), length=60)

            # get lstm predictions
            lstm_pred_scaled = lstm_models[ticker]['model'].predict(X_lstm_seq, verbose=0).flatten()
            lstm_pred_prices = lstm_models[ticker]['target_scaler'].inverse_transform(
                lstm_pred_scaled.reshape(-1, 1)
            ).flatten()

            # align lengths
            min_len = min(len(rf_proba), len(lstm_pred_prices))
            rf_proba = rf_proba[-min_len:]
            lstm_pred_prices = lstm_pred_prices[-min_len:]

            # get corresponding prices
            current_prices = data['Close'].values[-min_len-1:-1]
            future_prices = data['Close'].values[-min_len:]

            # generate ensemble signals
            signals = []
            for i in range(min_len):
                lstm_return = (lstm_pred_prices[i] - current_prices[i]) / current_prices[i]

                # both models must agree
                if rf_proba[i] > rf_threshold and lstm_return > lstm_threshold:
                    signals.append(1)
                else:
                    signals.append(0)

            strategies[ticker] = {
                'signals': np.array(signals),
                'rf_probabilities': rf_proba,
                'lstm_predictions': lstm_pred_prices,
                'current_prices': current_prices,
                'actual_prices': future_prices
            }

            signal_count = np.sum(signals)
            print(f"Generated {signal_count} BUY signals out of {len(signals)} periods")

        except Exception as e:
            print(f"Error generating predictions for {ticker}: {e}")

    return strategies

def calculate_rf_threshold(risk_score):
    # maps 0 - 100 score to 0.8 - 0.6 threshold range
    return round(0.80 - (risk_score / 100) * 0.20, 2)

def calculate_metrics(strategies, capital):
    metrics = {}

    # count tickers with buy signals before the loop
    tickers_with_signals = len([
        t for t in strategies
        if any(s == 1 for s in strategies[t]["signals"])
    ])

    for ticker, data in strategies.items():
        signals = data["signals"]
        current_prices = data["current_prices"]
        lstm_predictions = data["lstm_predictions"]

        # get buy signals only
        buy_indices = [i for i, s in enumerate(signals) if s == 1]

        if not buy_indices:
            continue

        # calculate returns for each buy signal
        trade_returns = []
        for i in buy_indices:
            predicted_return = (lstm_predictions[i] - current_prices[i]) / current_prices[i]
            trade_returns.append(predicted_return)

        trade_returns = np.array(trade_returns)

        # ROI
        roi = np.mean(trade_returns) * 100

        # Sharpe Ratio (assume risk free rate of 0)
        if np.std(trade_returns) > 0:
            sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(252)
        else:
            sharpe = 0

        # win rate
        wins = np.sum(trade_returns > 0)
        win_rate = (wins / len(trade_returns)) * 100

        # volatility
        volatility = np.std(trade_returns) * np.sqrt(252) * 100

        # equal allocation across tickers with buy signals
        allocation = capital / tickers_with_signals if tickers_with_signals > 0 else 0

        metrics[ticker] = {
            "roi": round(roi, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate": round(win_rate, 2),
            "volatility": round(volatility, 2),
            "buy_signals": len(buy_indices),
            "suggested_allocation": round(allocation, 2),
            "current_price": round(float(current_prices[-1]), 2),
            "predicted_price": round(float(lstm_predictions[-1]), 2),
        }

    return metrics

tickers = ['AAPL', 'GOOGL', 'NVDA', 'MSFT', 'TSLA', 'JPM', 'V', 'JNJ', 'AMZN', 'WMT']

# load on models on startup
rf_models = {}
lstm_models = {}

for ticker in tickers:
    try:
        rf_models[ticker] = {
            'model': joblib.load(f'models/rf_{ticker}.pkl'),
            'scaler': joblib.load(f'models/rf_scaler_{ticker}.pkl')
        }
        lstm_models[ticker] = {
            'model': keras.models.load_model(
                f'models/lstm_{ticker}.h5', 
                compile=False
            ),
            'feature_scaler': joblib.load(f'models/lstm_feature_scaler_{ticker}.pkl'),
            'target_scaler': joblib.load(f'models/lstm_target_scaler_{ticker}.pkl'),
        }
    
    except Exception as e:
        print(f"Error loading models for {ticker}: {e}")

news_cache = pd.DataFrame()

@app.on_event("startup")
async def startup():
    global news_cache
    sentiment = fetch_stock_news(tickers, days_back=30)
    news_cache = sentiment if sentiment is not None else pd.DataFrame()

class PredictRequest(BaseModel):
    capital: float
    returns: float
    profile: dict

@app.post("/predict")
def predict(req: PredictRequest, user=Depends(verify_token)):
    # fetch live data
    rf_stock_data = fetch_market_data(tickers, period="60d", interval="15m")
    lstm_stock_data = fetch_market_data(tickers, period="6mo", interval="1d")
    # get news from cache
    news_sentiment = news_cache

    # merge features
    rf_data = merge_data(rf_stock_data, news_sentiment, model="rf")
    lstm_data = merge_data(lstm_stock_data, news_sentiment, model="lstm")

    # combine rf and lstm features
    live_data = {}
    for ticker in tickers:
        if ticker in rf_data and ticker in lstm_data:
            combined = lstm_data[ticker].copy()
            for col in ['prev_open_close', 'prev_close_high', 'prev_close_low']:
                if col in rf_data[ticker].columns:
                    combined[col] = rf_data[ticker][col]
            live_data[ticker] = combined

    risk_score = calculate_risk(Profile(**req.profile))
    rf_threshold = calculate_rf_threshold(risk_score)
    lstm_threshold = req.returns / 100

    results = generate_strategies(rf_models, lstm_models, live_data, rf_threshold, lstm_threshold)
    
    strategies = {}
    for ticker, data in results.items():
        strategies[ticker] = {
            "signals": data["signals"].tolist(),
            "rf_probabilities": data["rf_probabilities"].tolist(),
            "lstm_predictions": data["lstm_predictions"].tolist(),
            "current_prices": data["current_prices"].tolist(),
            "actual_prices": data["actual_prices"].tolist(),
        }

    return {
        "strategies": strategies,
        "metrics": calculate_metrics(results, req.capital),
        "capital": req.capital,
        "expected_returns": req.returns,
        "risk_score": risk_score
    }