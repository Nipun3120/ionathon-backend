import asyncio
import json
import aiohttp
import requests
from typing import Optional
from urllib import request
import pandas as pd
from fastapi import FastAPI, Request
import numpy as np
import joblib
app = FastAPI()


import tensorflow as tf

import nest_asyncio
nest_asyncio.apply()

from datetime import datetime, timedelta
from helper import *

import twint
from scipy.stats import pearsonr

# twitter token Nipun3120
# consumerKey = "RqHmDknysI2pBPG9HczNTo1RD"
# consumerSecret = "ghkJPbD38xetBSJtrEruJrmJQTbhOHXKrxL8JzOSV5vkN4V2S7"
# accessToken = "776347800-qvsv6X2ow4n80QT6oQkDlMuFaqfx4ai6KkdBfZ6p"
# accessTokenSecret = "CwXy7hWwiGHVbpqveUCDBlg1UDp5o0ye6EsSY9xMtXZm9"


# MODEL_APPLE = tf.keras.models.load_model('models/apple_savedmodel/')

week_tweets = pd.DataFrame()

@app.get("/")
async def read_root(request: Request):
    # city, keyword
    data = await request.json()
    keyword = data['keyword'] 
    city = data['city']
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    today_60days = today - timedelta(days=60)
    yesterday_60days = yesterday - timedelta(days=60)
    today_80days = today - timedelta(days=90)
    
    
    # -------------- datetime ki dates today and yesturday ki date
    # today_res = await fetchStockPrices(today, today_80days, keyword)
    # yesterday_res = await fetchStockPrices(yesterday, today_80days, keyword)
     

    # stock_60_df = stock_60(today_res)
    # stock_yest_60_df = stock_60(yesterday_res)

    stock_60_df = await fetchStockPrices(today, today_80days, keyword)
    stock_yest_60_df = await fetchStockPrices(yesterday, today_80days, keyword)
     
    scaler, model = getRequiredScaler(keyword)


# ------- fetch sentiments mein dates pass krni hai 
    # tweets = await fetch_tweets(city, keyword, today_60days, today)
    global week_tweets
    week_tweets, pearson = await getTweetsWithSentiments(city, keyword, today-timedelta(days=10),today)        #check placement of this code

    
    sentiment_mean_today = week_tweets[week_tweets['date'] == str(today)]['sentiment'].mean()
    sentiment_mean_yest =  week_tweets[week_tweets['date'] == str(today-timedelta(days=1))]['sentiment'].mean()

    feature_array = np.append(scaler.fit_transform(np.array(stock_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_today)       #stop this from calling twice
    yest_feature_array = np.append(scaler.fit_transform(np.array(stock_yest_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_yest)  
    
    prediction = model.predict(feature_array.reshape(1,61,1))
    prediction_yest = model.predict(yest_feature_array.reshape(1,61,1))
    
    pred_inv_scaled = scaler.inverse_transform(prediction)
    pred_yest_inv_scaled = scaler.inverse_transform(prediction_yest)
    
    # print(float((pred_yest_inv_scaled - pred_inv_scaled)[0][0]))

    top = week_tweets[week_tweets ['date']== str(datetime.today().date())][['tweet', 'sentiment']].values.tolist()[:5]
    bottom = week_tweets[week_tweets['date'] == str(datetime.today().date())][['tweet', 'sentiment']].values.tolist()[-5:]

    # return float((pred_yest_inv_scaled - pred_inv_scaled)[0][0])
    corr, p_value = pearson
    return {"prediction":float((pred_yest_inv_scaled - pred_inv_scaled)[0][0]), "top": top, "bottom": bottom, "corr": corr, "p_value": p_value}


# for right screen
# this would be called from frontend as well, requirement: -> city and keyword, dates

@app.get("/search_tweet")
async def tweets_senti_for_front():
    city = "Cupertino"
    keyword = "AAPL"
    today = datetime.today().date()
    week_tweets, pearson = await getTweetsWithSentiments(city, keyword, today-timedelta(days=10),today)        #check placement of this code
    print(" ------ week_tweets -----", week_tweets)
    print(week_tweets[week_tweets['date'] == str(datetime.today().date())][['tweet', 'sentiment']])
    top = week_tweets[week_tweets ['date']== str(datetime.today().date())][['tweet', 'sentiment']].values.tolist()[:5]
    bottom = week_tweets[week_tweets['date'] == str(datetime.today().date())][['tweet', 'sentiment']].values.tolist()[-5:]
    print("-----> corr", pearson)
    print(top, bottom, type(top), type(bottom))
    return {'result':top + bottom}     #returns a list of lists with tweets and sentiments
# async def fetch_tweets(city, keyword, sinceDate, untilDate):

#     tweets = await getTweetsFromTwint(city, keyword, sinceDate, untilDate)
#     return tweets

# @app.get("/search_tweets/keywords")
# def fetch_sentiments(tweets):
    
#     tweet_sentiments = getSentimentsOfTweets(tweets)
#     return tweet_sentiments

# @app.get('/get-weekly-correlation')
# async def correlation(request: Request):
#     data = await request.json()

#     today = datetime.today().date()
#     ten_days = today - timedelta(days=10)

#     stock_price = await fetchStockPrices(today, ten_days, data['keyword'])
#     week_tweets

#     print(stock_price)
#     s = set(list(stock_price['Date']))
#     n = set(list(week_tweets['Date']))

#     common_dates_1 = [x for x in s if x in n]

#     stock_price = stock_price.loc[stock_price['Date'].isin(common_dates_1)].reset_index()
#     week_tweet1 = week_tweets.loc[week_tweets['date'].isin(common_dates_1)].reset_index().copy()

#     stock_price['open-close'] = stock_price['Open'] - stock_price['Close']

#     return pearsonr(stock_price['open-close'], week_tweet1['sentiment'])
    