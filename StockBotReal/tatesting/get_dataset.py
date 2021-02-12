from iexfinance.stocks import get_historical_data
#import iexfinance
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
import os
import pandas as pd
import apikeys
import datetime as dt
#export IEX_TOKEN=6e087e35a8544a10ae7f8c4aad404854
api_key = apikeys.FINN_API_KEY
stock = 'SQ'
resolution = 'D'
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365)
end = int(end_date.timestamp())
start = int(start_date.timestamp())

def get_historical_data(symbol):
    ts = TimeSeries(key="55S0MXQGMRA9RH5C", output_format='pandas')
    #finnhub_client = finnhub.Client(api_key=api_key)
    #result = finnhub_client.stock_candles(stock, resolution, start, end)
    #os.environ["IEX_TOKEN"] = "6e087e35a8544a10ae7f8c4aad404854"
    #result = get_historical_data("AAPL", start="20170101", end="20180101", output_format='pandas').head()
    result = ts.get_intraday(symbol, "1min", 'full')
    print(result[0])
    df = pd.DataFrame(result[0])
    print(df)
    #print(df["1. open"])
    for col in df.columns:
        print("columnName:"+str(col))
#    print(df.loc[["2020-12-24"]])
    cols = ['1. open', '2. high', '3. low', '4. close', '4. close', '5. volume']

    df = df[cols]
    df.columns = ['Open','High','Low','Close','Adj. Close','Volume']



    df = df.iloc[::-1]
    #Date,Open,High,Low,Close,Adj Close,Volume
    print(df)
    # for (columnName, columnData) in df.iteritems():
    #     #print('Colunm Name : ', columnName)
    #     # print('Column Contents : ', columnData.values)
    #     if columnName == 'Date':
    #         for value in columnData.values:
    #             new = dt.datetime.fromtimestamp(value)
    #             new = str(new)[:-9]
    #             #print(new)
    #             df['Date'].replace({value: new}, inplace=True)
    #         #print('Column Contents : ', columnData.values)

    #df.to_csv("./{}.csv".format(stock), index=False)
    return df

if __name__== "main":
    get_historical_data()


#/home/ryan/IdeaProjects/StockBotReal/stockpredictiontesting/Stock-Prediction-Models/dataset/AMD2020.csv