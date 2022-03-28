import pandas as pd
from fastapi import FastAPI, Request
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import tensorflow as tf

import nest_asyncio
nest_asyncio.apply()

from datetime import datetime, timedelta
from helper import *


# week_tweets_djia = pd.DataFrame()
# DJIA
@app.get("/fetch-data/djia-data")
async def read_root_djia():
    # city, keyword
    # data = await request.json()
    keyword = "DJIA"
    city = "New York"
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    today_60days = today - timedelta(days=60)
    yesterday_60days = yesterday - timedelta(days=60)
    today_80days = today - timedelta(days=90)
    
    stock_60_df = await fetchStockPrices(today, today_80days, keyword)
    stock_yest_60_df = await fetchStockPrices(yesterday, today_80days, keyword)
     
    scaler, model = getRequiredScaler(keyword)

    # global week_tweets_djia
    week_tweets_djia, pearson = await getTweetsWithSentiments(city, keyword, today-timedelta(days=10),today)        #check placement of this code
    
    if str(week_tweets_djia['date'][0]) == str(today-timedelta(days=1)):       #if today tweet not available
        print('date condition--->',str(week_tweets_djia['date'][0]),str(today-timedelta(days=1)))
        today = today-timedelta(days=1)
    # exit()
    sentiment_mean_today = week_tweets_djia[week_tweets_djia['date'] == str(today)]['sentiment'].mean()
    sentiment_mean_yest =  week_tweets_djia[week_tweets_djia['date'] == str(today-timedelta(days=1))]['sentiment'].mean()
    print('sentiment-->', sentiment_mean_today, sentiment_mean_yest)
    # exit()
    feature_array = np.append(scaler.fit_transform(np.array(stock_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_today)       #stop this from calling twice
    yest_feature_array = np.append(scaler.fit_transform(np.array(stock_yest_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_yest)  

    
    prediction = model.predict(feature_array.reshape(1,61,1))
    prediction_yest = model.predict(yest_feature_array.reshape(1,61,1))
    
    pred_inv_scaled = scaler.inverse_transform(prediction)
    pred_yest_inv_scaled = scaler.inverse_transform(prediction_yest)
    

    top = week_tweets_djia[week_tweets_djia ['date']== str(today)][['tweet', 'sentiment']].values.tolist()[:5]
    bottom = week_tweets_djia[week_tweets_djia['date'] == str(today)][['tweet', 'sentiment']].values.tolist()[-5:]

    corr, p_value = pearson

    return {"prediction_djia":float((pred_yest_inv_scaled - pred_inv_scaled)[0][0]), "top": top+bottom, "corr": corr, "p_value": p_value, "prices":list(stock_60_df['Close'])}



# week_tweets_nifty = pd.DataFrame()
# NIFTY
@app.get("/fetch-data/nifty-data")
async def read_root_nifty():
    # city, keyword
    # data = await request.json()
    keyword = "NIFTY"
    city = "Mumbai"
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    today_60days = today - timedelta(days=60)
    yesterday_60days = yesterday - timedelta(days=60)
    today_80days = today - timedelta(days=90)
    
    stock_60_df = await fetchStockPrices(today, today_80days, keyword)
    stock_yest_60_df = await fetchStockPrices(yesterday, today_80days, keyword)
     
    scaler, model = getRequiredScaler(keyword)

    # global week_tweets_nifty
    week_tweets_nifty, pearson = await getTweetsWithSentiments(city, keyword, today-timedelta(days=10),today)        #check placement of this code
    
    if str(week_tweets_nifty['date'][0]) == str(today-timedelta(days=1)):      #if today tweet not available
        today = today-timedelta(days=1)

    sentiment_mean_today = week_tweets_nifty[week_tweets_nifty['date'] == str(today)]['sentiment'].mean()
    sentiment_mean_yest =  week_tweets_nifty[week_tweets_nifty['date'] == str(today-timedelta(days=1))]['sentiment'].mean()

    feature_array = np.append(scaler.fit_transform(np.array(stock_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_today)       #stop this from calling twice
    yest_feature_array = np.append(scaler.fit_transform(np.array(stock_yest_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_yest)  
    
    prediction = model.predict(feature_array.reshape(1,61,1))
    prediction_yest = model.predict(yest_feature_array.reshape(1,61,1))
    
    pred_inv_scaled = scaler.inverse_transform(prediction)
    pred_yest_inv_scaled = scaler.inverse_transform(prediction_yest)
    

    top = week_tweets_nifty[week_tweets_nifty ['date']== str(today)][['tweet', 'sentiment']].values.tolist()[:5]
    bottom = week_tweets_nifty[week_tweets_nifty['date'] == str(today)][['tweet', 'sentiment']].values.tolist()[-5:]

    corr, p_value = pearson

    counttop = 0

    return {"prediction_nifty":float((pred_yest_inv_scaled - pred_inv_scaled)[0][0]), "top": top+bottom, "corr": corr, "p_value": p_value, "prices":list(stock_60_df['Close'])}



# # week_tweets_apple = pd.DataFrame()
# # AAPL
@app.get("/fetch-data/apple-data")
async def read_root_apple():
    # city, keyword
    # data = await request.json()
    keyword = "AAPL"
    city = "New York"
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    today_60days = today - timedelta(days=60)
    yesterday_60days = yesterday - timedelta(days=60)
    today_80days = today - timedelta(days=90)
    
    stock_60_df = await fetchStockPrices(today, today_80days, keyword)
    stock_yest_60_df = await fetchStockPrices(yesterday, today_80days, keyword)
     
    scaler, model = getRequiredScaler(keyword)

    # global week_tweets_apple
    week_tweets_apple, pearson = await getTweetsWithSentiments(city, keyword, today-timedelta(days=10),today)        #check placement of this code
    
    if str(week_tweets_apple['date'][0]) == str(today-timedelta(days=1)):      #if today tweet not available
        today = today-timedelta(days=1)

    # today = today-timedelta(days=1)

    sentiment_mean_today = week_tweets_apple[week_tweets_apple['date'] == str(today)]['sentiment'].mean()
    sentiment_mean_yest =  week_tweets_apple[week_tweets_apple['date'] == str(today-timedelta(days=1))]['sentiment'].mean()


    feature_array = np.append(scaler.fit_transform(np.array(stock_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_today)       #stop this from calling twice
    yest_feature_array = np.append(scaler.fit_transform(np.array(stock_yest_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_yest)  
    
    prediction = model.predict(feature_array.reshape(1,61,1))
    prediction_yest = model.predict(yest_feature_array.reshape(1,61,1))
    
    pred_inv_scaled = scaler.inverse_transform(prediction)
    pred_yest_inv_scaled = scaler.inverse_transform(prediction_yest)
    

    top = week_tweets_apple[week_tweets_apple ['date']== str(today)][['tweet', 'sentiment']].values.tolist()[:5]
    bottom = week_tweets_apple[week_tweets_apple['date'] == str(today)][['tweet', 'sentiment']].values.tolist()[-5:]

    corr, p_value = pearson

    return {"prediction_apple":float((pred_yest_inv_scaled - pred_inv_scaled)[0][0]), "top": top+bottom, "corr": corr, "p_value": p_value, "prices":list(stock_60_df['Close'])}





# week_tweets_micro = pd.DataFrame()
#MSFT
@app.get("/fetch-data/microsoft-data")
async def read_root_micro():
    # city, keyword
    # data = await request.json()
    keyword = "MSFT"
    city = "New York"
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    today_60days = today - timedelta(days=60)
    yesterday_60days = yesterday - timedelta(days=60)
    today_80days = today - timedelta(days=90)
    
    stock_60_df = await fetchStockPrices(today, today_80days, keyword)
    stock_yest_60_df = await fetchStockPrices(yesterday, today_80days, keyword)
     
    scaler, model = getRequiredScaler(keyword)

    # global week_tweets_micro
    week_tweets_micro, pearson = await getTweetsWithSentiments(city, keyword, today-timedelta(days=10),today)        #check placement of this code

    if str(week_tweets_micro['date'][0]) == str(today-timedelta(days=1)):      #if today tweet not available
        today = today-timedelta(days=1)

    print('date-----> ', week_tweets_micro['date'][0])
    
    sentiment_mean_today = week_tweets_micro[week_tweets_micro['date'] == str(today)]['sentiment'].mean()
    sentiment_mean_yest =  week_tweets_micro[week_tweets_micro['date'] == str(today-timedelta(days=1))]['sentiment'].mean()

    feature_array = np.append(scaler.fit_transform(np.array(stock_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_today)       #stop this from calling twice
    yest_feature_array = np.append(scaler.fit_transform(np.array(stock_yest_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_yest)  
    
    prediction = model.predict(feature_array.reshape(1,61,1))
    prediction_yest = model.predict(yest_feature_array.reshape(1,61,1))
    
    pred_inv_scaled = scaler.inverse_transform(prediction)
    pred_yest_inv_scaled = scaler.inverse_transform(prediction_yest)
    

    top = week_tweets_micro[week_tweets_micro ['date']== str(today)][['tweet', 'sentiment']].values.tolist()[:5]
    bottom = week_tweets_micro[week_tweets_micro['date'] == str(today)][['tweet', 'sentiment']].values.tolist()[-5:]

    corr, p_value = pearson

    return {"prediction_microsoft":float((pred_yest_inv_scaled - pred_inv_scaled)[0][0]), "top": top+bottom, "corr": corr, "p_value": p_value, "prices":list(stock_60_df['Close'])}



