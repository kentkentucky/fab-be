import pandas as pd
import numpy as np
import ta

# function to engineer specific features for random forest
def engineer_rf_features(data):
    # copy data to data frame
    df = data.copy()

    # create target variable
    df['future_returns'] = df['Close'].pct_change().shift(-1)
    df['signal'] = (df['future_returns'] > 0).astype(int)

    # get specific features
    df['rsi'] = ta.momentum.RSIIndicator(
        close=df['Close'],
        window=14
    ).rsi()
    df['sma_50'] = ta.trend.SMAIndicator(
        close=df['Close'],
        window=50
    ).sma_indicator()
    df['adx'] = ta.trend.ADXIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=20
    ).adx()
    df['volume'] = df['Volume']
    df['corr'] = df['Close'].rolling(window=24).corr(df['Volume'])
    df['prev_open_close'] = (df['Open'] - df['Close']).shift(1)
    df['prev_close_high'] = (df['Close'] - df['High']).shift(1)
    df['prev_close_low'] = (df['Close'] - df['Low']).shift(1)
    df['momentum'] = df['Close'].diff(20)
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

    # return data frame
    return df

# function to engineer specific features for lstm
def engineer_lstm_features(data):
    # copy data to data frame
    df = data.copy()

    # get specific features
    df['rsi'] = ta.momentum.RSIIndicator(
        close=df['Close'],
        window=14
    ).rsi()
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    df['macd'] = ta.trend.MACD(
        close=df['Close']
    ).macd()
    df['volume'] = df['Volume']
    df['adx'] = ta.trend.ADXIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=20
    ).adx()
    df['sma_50'] = ta.trend.SMAIndicator(
        close=df['Close'],
        window=50
    ).sma_indicator()
    df['momentum'] = df['Close'].diff(20)
    df['corr'] = df['Close'].rolling(window=24).corr(df['Volume'])
    df['target'] = df['Close'].shift(-1)

    # return data frame
    return df

# function to merge stock data and sentiment data
def merge_technical_sentiment(technical_df, sentiment_df):
    # reset data frame index
    technical_df = technical_df.reset_index()

    # check for date in lstm data
    if 'Date' in technical_df.columns:
        date_col = 'Date'
    # check for datetime in rf data
    elif 'Datetime' in technical_df.columns:
        date_col = 'Datetime'
    else:
        raise ValueError(f"No date column found.")

    # convert date_col column into datetime
    technical_df['date'] = pd.to_datetime(technical_df[date_col]).dt.date

    if not sentiment_df.empty:
         # convert date column into datetime
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date

        # merge stock data and sentiment data based on date column
        merged = technical_df.merge(
            sentiment_df,
            on='date',
            how='left'
        )
    else:
        # default score if no sentiment data
        merged = technical_df
        merged['sentiment_mean'] = 0
        merged['sentiment_std'] = 0
        merged['news_count'] = 0

    # replace na values to 0
    merged['sentiment_mean'] = merged['sentiment_mean'].fillna(0)
    merged['sentiment_std'] = merged['sentiment_std'].fillna(0)
    merged['news_count'] = merged['news_count'].fillna(0)

    # get absolute value of sentiment mean
    merged['sentiment_strength'] = merged['sentiment_mean'].abs()
    # get sentiment volume by product of sentiment mean and news count
    merged['sentiment_volume'] = merged['sentiment_mean'] * merged['news_count']

    # return merged data frame
    return merged

# function to merge market data and sentiment data based on model
def merge_data(stock_data, news_sentiment, model):
    # get number of tickers that are unique
    tickers = stock_data['ticker'].unique()
    # initialise final data object
    final_data = {}

    # loop through tickers
    for ticker in tickers:
        # get copy of stock data according to ticker
        ticker_stock_data = stock_data[stock_data['ticker'] == ticker].copy()
        # get copy of sentiment data according to ticker
        ticker_sentiment = news_sentiment[news_sentiment['ticker'] == ticker].copy()

        # engineer features based on model
        if model == "rf":
            processed = engineer_rf_features(ticker_stock_data)
        elif model == "lstm":
            processed = engineer_lstm_features(ticker_stock_data)
        else:
            raise ValueError(f"Invalid model: {model}.")

        # check for data in ticker sentiment data frame
        if not ticker_sentiment.empty:
            # merge market data and sentiment data
            merged = merge_technical_sentiment(processed, ticker_sentiment)
        else:
            merged = merge_technical_sentiment(processed, pd.DataFrame())

        # save data into object
        final_data[ticker] = merged

    # return final data
    return final_data