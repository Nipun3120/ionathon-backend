import requests
import os
import json
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import date
import numpy as np
import joblib
import twint

from scipy.stats import pearsonr
from flair.models import TextClassifier
from flair.data import Sentence
import tensorflow as tf
from nsepy import get_history
from datetime import datetime, timedelta
# import datetime
import yfinance as yf

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
# bearer_token = os.environ.get("BEARER_TOKEN")

MODEL_APPLE = tf.keras.models.load_model('models/apple_savedmodel/')
MODEL_MICROSOFT = tf.keras.models.load_model('models/microsoft_savedmodel/')
MODEL_NIFTY = tf.keras.models.load_model('models/nifty_savedmodel/')
MODEL_DOW = tf.keras.models.load_model('models/dow_savedmodel/')


search_url = "https://api.twitter.com/2/tweets/search/recent"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
query_params = {'query': '(from:twitterdev) OR #AAPL'}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer AAAAAAAAAAAAAAAAAAAAAAgMagEAAAAA1jhzo0ae0HAaHTu82gDkAAntQFA%3DeMwc9XfLQ3N77crjOs1YUva7GZoJ34Afl1DW85EJPPJMYj1HhW"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params, day):
    response = requests.get(url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


# def fetch_tweets_from_api(day):

#     json_response = connect_to_endpoint(search_url, query_params, day)
# #     print(json.dumps(json_response, indent=4, sort_keys=True))
#     return json_response


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def clean(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'pic.twitter\S+', ' ', text)
    # text = decontracted(text)
    text = re.sub(r'\(([^)]+)\)', " ", text)
    text = text.replace('etmarkets', ' ').replace('marketupdates', ' ').replace('newsalert', ' ').replace('ndtv', ' ').replace('moneycontrol', ' ').replace('here is why', ' ')
    text = text.replace('marketsupdate', ' ').replace('biznews', ' ').replace('click here', ' ').replace('live updates', ' ').replace('et now', ' ')
    text = re.sub(r'[^a-zA-Z0-9 ]+', ' ', text)
    text = re.sub(r' \w{1,2}_', ' ', text)
    text = re.sub('\s+',' ', text)
    return text


analyzer = SentimentIntensityAnalyzer()
def sentiments(text):
    return analyzer.polarity_scores(text)['compound']

classifier = TextClassifier.load('en-sentiment')
def flair_sentiments(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    return sentence.labels[0].score if sentence.labels[0].value == 'POSITIVE' else -sentence.labels[0].score

def getSentimentsOfTweets(tweets):
    return [flair_sentiments(x) for x in tweets]

def weekly_correlation(df):
    a = np.array(df['close_price'])
    b = np.array(df['score'])
    b_adjusted = np.append(b[1:],1)
    # b_adjusted, b

    return pearsonr(a,b_adjusted)

def stripdate(date):
    return date.split(' ')[0]
# abbe krne de do min acha oki
# change all occurences kara hai
#thode changes karne padenge, isme sab kaam ho rha ab, short mein. tumhara front end pe bhi yahi se bhej dena
async def getTweetsWithSentiments(city, keyword, sinceDate, untilDate):
    config = twint.Config()
    config.Near = city
    config.Search = keyword
    config.Lang = "en"
    config.Since = str(sinceDate)
    config.Until = str(untilDate)
    # config.Until = str(untilDate.date())
    config.Popular_tweets = True
    config.Pandas = True
    config.limit = 100

    twint.run.Search(config)
    tweets_df = twint.storage.panda.Tweets_df
    # tweets = list(tweets_df['tweet'])

    tweets_df['date'] = tweets_df['date'].apply(stripdate)
    tweets_df['tweet_clean'] = tweets_df['tweet'].apply(clean)
    tweets_df = tweets_df.sort_values(['date','nreplies','nlikes','nretweets'], ascending=[False, False, False, False])[['date', 'tweet', 'tweet_clean', 'nlikes','nretweets', 'nreplies']].reset_index()

    a = list(tweets_df.columns)
    a.append('sentiment')
    df_final = pd.DataFrame(columns = a)
    for date in list(tweets_df['date'].unique()):
        df_temp = pd.DataFrame(columns = tweets_df.columns)
        df_temp = df_temp.append(tweets_df[tweets_df['date']==date][:20])

        df_temp['sentiment'] = tweets_df[tweets_df['date']==date][:20]['tweet'].apply(flair_sentiments)
        df_final = df_final.append(df_temp)
    df_final = df_final.sort_values(['date', 'sentiment'], ascending= [False, False])

    dates = []
    prices = []

    stock_prices = await fetchStockPrices(untilDate, untilDate - timedelta(days=10), keyword)
    stock_prices['open-close'] = stock_prices ['Open'] - stock_prices['Close']

    print(stock_prices)
    print(df_final)

    s = set(list(df_final['date']))
    n = set([x for x in list(stock_prices['Date'])])

    print(n, s)

    common_dates_1 = [x for x in s if x in n]

    print(common_dates_1)

    for i in common_dates_1:
        # print(i)
        # print('sentiment mean', type(df_final[df_final['date'] == i]['sentiment'].mean()),df_final[df_final['date'] == i]['sentiment'].mean())
        dates.append(df_final[df_final['date'] == i]['sentiment'].mean())
        prices.append(stock_prices[stock_prices['Date'] == i]['open-close'].values[0])

    
    print('--------->', dates)
    print('---------->', prices)
    

    return df_final, pearsonr(np.append(np.array(prices)[1:], 1), np.array(dates))


def getTopTweets(tweets, sentiments):
    df = pd.DataFrame(list(zip(tweets, sentiments)), columns=['tweets', 'sentiments'])
    df = df.sort_values('sentiments')
    return df

def correct_date_fmt(date):
    return str(date.strftime('%Y-%m-%d'))
# returns a list of prices (okay?)
async def fetchStockPrices(currDate, prevDate, keyword):
    
    if keyword == "MSFT" or keyword == "AAPL":
        res = requests.get(f'https://api.polygon.io/v2/aggs/ticker/{keyword}/range/1/day/{prevDate}/{currDate}?adjusted=true&sort=desc&apiKey=39ngr6sQIbZKPDfWGh4yMh5dLzpL9wzf').json()

        # datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
        stock_60 = pd.DataFrame(list(zip([datetime.fromtimestamp(agg['t']/1000).strftime('%Y-%m-%d') for agg in res['results']][:60], [agg['c'] for agg in res['results']][:60], [agg['o'] for agg in res['results']][:60])), columns = ['Date', 'Close', 'Open'])
        # stock_60 = {'Close':[agg['c'] for agg in res['results']][:60], 'Open':[agg['c'] for agg in res['results']][:60] }
        return stock_60

    if keyword == "NIFTY":
        print(prevDate, currDate)
        data = get_history(symbol=keyword, start=prevDate, end=currDate, index=True)
        # print(data)
        a = data.index.copy()
        data = data.reset_index()
        data['Date'] = a
        data['Date'] = data['Date'].apply(correct_date_fmt)
        # print(data)
        return data[['Date','Close','Open']][:60]
        
    if keyword == "DJIA":
        dow = yf.Ticker("^DJI")
        res = dow.history(period='1d', start=str(prevDate), end=str(currDate))

        a = res.index.copy()
        res = res.reset_index()
        res['Date'] = a
        res['Date'] = res['Date'].apply(correct_date_fmt)
        
        return res[['Date', 'Close', 'Open']][:60]


def stock_60(stockArr):
    stock_60 = [agg['c'] for agg in stockArr['results']]
    return stock_60[:60]

def getRequiredScaler(keyword):
    if keyword == "AAPL":
        scaler = joblib.load('scalers/apple_scaler.save')
        model = MODEL_APPLE
        return {scaler: scaler, model: model}

    elif keyword == "MSFT":
        scaler = joblib.load('scalers/microsoft_scaler.save')
        model = MODEL_MICROSOFT
        return {scaler: scaler, model: model}

    elif keyword == "DJIA":
        scaler = joblib.load('scalers/dow_scaler.save')
        model = MODEL_DOW
        return {scaler: scaler, model: model}

    elif keyword == "NIFTY":
        scaler = joblib.load('scalers/nifty_scaler.save')
        model = MODEL_NIFTY
        return {scaler: scaler, model: model}
        


        
