from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


#import alpaca_backtrader_api as Alpaca

import time

import pandas

import talib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import argrelextrema
from talib import MA_Type
import ta
import re


import apikeys as alpaca_paper

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt


# Create a Stratey








#ALPACA_KEY_ID = alpaca_paper['API_KEY']
#ALPACA_SECRET_KEY = alpaca_paper['API_SECRET']
#ALPACA_PAPER = True


shortPeriod = 14
longPeriod = 200


def SMASig(close, sPeriod, lPeriod):
    shortSMA = ta.SMA(close, sPeriod)
    longSMA = ta.SMA(close, lPeriod)
    smaSell = ((shortSMA <= longSMA) & (shortSMA.shift(1) >= longSMA.shift(1)))
    smaBuy = ((shortSMA >= longSMA) & (shortSMA.shift(1) <= longSMA.shift(1)))
    return smaSell, smaBuy, shortSMA, longSMA


def dropna(arr, *args, **kwarg):
    assert isinstance(arr, np.ndarray)
    dropped = pandas.DataFrame(arr).dropna(*args, **kwarg).values
    if arr.ndim == 1:
        dropped = dropped.flatten()
    return dropped


def getSignals():
    maxs = argrelextrema(ams, np.greater)

    #print(str(maxs[0]))
    maxs1 = str(maxs[0])[2:][:-1].split(" ")
    # maxs1.remove('')
    for index, val in enumerate(maxs1):
        if "\n" in val:
            print(val)
            val = val.replace('\n', "")
            maxs1[index] = val
    #print(maxs1)
    # print(maxs[0][0])
    maxs11 = filtersignals(maxs1)
    maxsnp = np.array(maxs11)
    maxs11 = dropna(maxsnp)
    # maxs11 = maxs11[np.logical_not(np.isnan(maxs11))]
    maxs2 = []
    for x in maxs11:
        maxs2.append(int(x))
    #print(maxs2)
    #decidesignals(maxs2)
    decidesignals2(maxs2)
    maxs2 = np.array(maxs2)
    #print(maxs)
    ymaxs = []
    #print(np.size(maxs2))
    for x in maxs2:
        #print(int(x))
        ymaxs.append(ams[x])
    # axsnp = np.array(ymaxs)
    #print(ymaxs)
    filter(lambda v: v == v, ymaxs)

    #print(ymaxs)
    return maxs2, ymaxs



def findDateLine(date, stock):
    print(date+stock)
    #x= re.search(r'{0}:\w\w\w'.format(date))
    dates = df['date']
    print(dates)
    print(dates.values)
    for index, row in enumerate(dates.values):
        if date in row:
            print("index")
            return index

    return 1111

def filtersignals(signals):
    global actionlist
    # signals = signals[0]
    # print(type(signals))
    prev = 0
    while ('' in signals):
        signals.remove('')
    for index, sig in enumerate(signals):
        print(str(sig) + ", " + str(prev))
        # print(index)
        try:
            sig = int(sig)
        except:
            print("bad")
            maxs1.remove(sig)
            continue
        try:
            futureone = signals[index + 1]
            if abs(sig - prev) <= 20 or abs(int(futureone) - sig) <= 20:
                print("too close: " + str(sig) + ", " + str(prev))
                try:
                    print("remove")
                    signals.remove(sig)
                except:
                    try:
                        del signals[index]
                    except:

                        print("cant remove: " + str(sig))
                continue
        except Exception as e:
            print(e)
            print("index error")
            continue
        print(sig)

        prev = sig
    # print("final actrion list")
    # print(finalactionlist)
    return signals


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom;
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

    return A, B, C


def parabolavertex(a, b, c):
    vertex = [(-b / (2 * a)), (((4 * a * c) - (b * b)) / (4 * a))]
    # print("Vertex: (", (-b / (2 * a)), ", "
    #       , (((4 * a * c) - (b * b)) / (4 * a)), ")")

    # print("Focus: (", (-b / (2 * a)), ", "
    #       , (((4 * a * c) - (b * b) + 1) / (4 * a)), ")")
    #
    # print("Directrix: y="
    #       , (int)(c - ((b * b) + 1) * 4 * a))

    return vertex


def filterParabolaSignals():
    global finalactionlist
    #print("filtering parabolas")
    print(parabolas)
    for index, val in enumerate(parabolas):
        print(val)
        plusthreshold = 0.000105
        plusthresholdmax = 5
        negthreshold = -0.000105
        negthresholdmax = -5
        if plusthreshold < float(parabolas[val][0]) < plusthresholdmax:
            print("Good VAL Up")
            print(str(plusthreshold) + "<" + str(float(parabolas[val][0])) + "<" + str(plusthresholdmax))
            goodparabolas[val] = parabolas[val]
            finalactionlist[val]="BUY"
        else:
            if negthreshold > parabolas[val][0] > negthresholdmax:
                print("good down")
                goodparabolas[val] = parabolas[val]
                finalactionlist[val]="SELL"
            else:
                print(str(val) + " not within threshhold removing")
                # parabolas.pop(val)


def decidesignals2(signals):
    global parabolas

    #print(signals)
    for index, val in enumerate(signals):

        print(index)
        print(val)
        print("parabolas pressent")
        #print(parabolas)
        if val in parabolas:
            print("already made parabola")
            continue
        # if val <= 6:
        #     continue

        print("parabola deciding")
        print(val)
        print("#################################")
        # print(ams)
        y = ams[val - 25:]
        y = y[~np.isnan(y)]
        # y = y[:-1*(len(y)-val-5)]
        print(y)
        y = y.tolist()
        filter(lambda v: v == v, y)
        x = np.linspace(val - 25, val + 5, len(y))
        filter(lambda v: v == v, x)
        # print(len(y))
        #print(x)
        #print(y)

        print("#################################")
        try:
            fit_params, pcov = scipy.optimize.curve_fit(parabola, x, y)
        except:
            print("NNOOOOOOOO")
            continue
        parabolas[val] = fit_params

    print(parabolas)
    return parabolas


hist = []
count = 0
actlist = []
global actionlist
global finalactionlist
global parabolas
global plottedparabolas
global goodparabolas
finalactionlist = {}
plottedparabolas = {}
parabolas = {}
goodparabolas = {}


global sma
global ema
global testma
global ams












class TestStrategy(bt.Strategy):
    global sma
    global ema
    global testma
    global ams
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(self.data, period=50)
        # To keep track of pending orders
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None


    def next(self):
        global sma
        global ema
        global testma
        global ams

# Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        cumulativeprices = list(self.datas[0])
        # print(index)
        #index = index + dateline
        #row1 = fd['Close']
        # print(row1)
        prices = cumulativeprices
        # print(prices)
        if len(cumulativeprices) > 5:
            sma = talib.SMA(np.array(cumulativeprices), timeperiod=30)
            #sma = self.sma[0]
            #print("SMAAAAA")
            #print(sma)
            #ema = talib.EMA(np.array(cumulativeprices), timeperiod=30)
            #testma = talib.SMA(np.array(cumulativeprices), timeperiod=15)
            #ams = np.roll(sma, -18)
            #ams = sma
            ams = np.array(sma)
            xmaxs, ymaxs = getSignals()


        if len(parabolas) >= 1:
            print("plotting parobalsf")
           # print(parabolas)
            print(goodparabolas)
            filterParabolaSignals()
            for index, val in enumerate(goodparabolas):
                print(val)
                #print(plottedparabolas.keys())

                # d = list(val)

                fit_params = parabolas[val]
                # fit_params = val[]
                parvertext = parabolavertex(*fit_params)
                #print(parvertext)
                if goodparabolas[val][0] < 0 and val not in plottedparabolas.keys():
                    print("sell")

                    # SELL, SELL, SELL!!! (with all possible default parameters)
                    self.log('SELL CREATE, %.2f' % self.dataclose[0])
                    print("sell: " + str(val))
                    # Keep track of the created order to avoid a 2nd order
                    if self.position:
                        self.order = self.sell()
                        self.order = self.sell()
                    else:
                        self.order = self.sell()
                if goodparabolas[val][0] > 0 and val not in plottedparabolas.keys():
                    print("BUY")

                    self.log('BUY CREATE, %.2f' % self.dataclose[0])
                    print("buy: " + str(val))
                    # Keep track of the created order to avoid a 2nd order

                    if self.position:
                        self.order = self.buy()
                        self.order = self.buy()
                    else:
                        self.order = self.buy()



                plottedparabolas[val] = True















if __name__ == '__main__':
    #global ams
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, './datasets/IBM.csv')

    # Create a Data Feed
    data = bt.feeds.GenericCSVData(
        dataname=datapath,
        timeframe=bt.TimeFrame.Minutes,
        compression=1, # The data is already at 5 minute intervals
        fromdate=datetime.datetime(2021, 1, 5, 9, 30),
        todate=datetime.datetime(2021, 1,6, 16, 30),
        sessionstart=datetime.time(9, 30),
        sessionend=datetime.time(16, 30),
        dtformat='%Y-%m-%d %H:%M:%S',
        datetime=0,
        time=-1,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=6,
        openinterest=-1,
        headers=1,
        separator=",",
        reverse=True
    )
    data.addfilter(bt.filters.SessionFilter(data))


    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(25000.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()





