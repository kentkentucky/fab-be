from massive import RESTClient
from dotenv import load_dotenv
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import os
import time

load_dotenv()

# load api key from .env
news_api_key = os.getenv("NEWS_API_KEY")
client = RESTClient(news_api_key)

# function to get recent market data
def fetch_market_data(tickers, period, interval):
    # initialise data
    data = []

    # loop through tickers
    for ticker in tickers:
        try:
            # get data according to ticker, period and interval
            df = yf.download(ticker, period=period, interval=interval, progress=False)

            # check for data 
            if len(df) > 0:
                # check for multi index
                if isinstance(df.columns, pd.MultiIndex):
                    # flatten multi index column
                    df.columns = df.columns.get_level_values(0)

                # reset index
                df = df.reset_index()
                # add ticker column
                df['ticker'] = ticker
                # append into data 
                data.append(df)

            else:
                # no data available
                print(f"No data")
        except Exception as e:
            # error message
            print(f"Error fetching data for {ticker}: {e}")

    # concat data array into single dataframe
    return pd.concat(data, ignore_index=True)

# function to fetch recent stock news
def fetch_stock_news(tickers, days_back=30):
    # initialise news
    news = []

    # loop through tickers
    for i, ticker in enumerate(tickers):
        try:
            # get current date time
            to_date = datetime.now()
            # calculate date of when you want to get the news from
            from_date = to_date - timedelta(days=days_back)

            # intialise news item array
            news_items = []
            # get news item from api
            for n in client.list_ticker_news(
                ticker=ticker,
                published_utc_gte=from_date.strftime('%Y-%m-%d'),
                published_utc_lte=to_date.strftime('%Y-%m-%d'),
                order="desc",
                limit=100,
                sort="published_utc"
            ):
                # append news into news item array
                news_items.append(n)

            if not news_items:
                print(f"{ticker}: No news found")
                continue

            # map sentiment to values
            sentiment_map = {
                "positive": 1,
                "neutral": 0,
                "negative": -1
            }

            # loop through news item
            for article in news_items:
                # initialise sentiment to 0
                sentiment = 0

                # check for insights in article
                if article.insights:
                    # loop through insight
                    for insight in article.insights:
                        # get sentiment for ticker
                        if insight.ticker == ticker:
                            # get sentiment score using sentiment map
                            sentiment = sentiment_map.get(insight.sentiment, 0)
                            # break out of loop
                            break

                # append object into news data array
                news.append({
                    "ticker": ticker,
                    "date": pd.to_datetime(article.published_utc).date(),
                    "title": article.title,
                    "sentiment": sentiment,
                    "publisher": article.publisher.name if article.publisher else 'Unknown'
                })

        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit for {ticker}, waiting 60s...")
                time.sleep(60)
            else:
                # print error
                print(f"Error fetching news for {ticker}: {e}")

        # rate limit buffer between tickers
        if i < len(tickers) - 1:
            print(f"Waiting 90 seconds before next ticker...")
            time.sleep(90)
    
    if not news:
        print("No news data to save.")
        return
    
    df = pd.DataFrame(news)
    daily_sentiment = df.groupby(['ticker', 'date']).agg(
        sentiment_mean=('sentiment', 'mean'),
        sentiment_std=('sentiment', 'std'),
        news_count=('sentiment', 'count')
    ).reset_index()

    return daily_sentiment