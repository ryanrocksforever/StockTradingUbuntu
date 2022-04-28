# date 3-23-22
import datetime
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
import logging
import sys
# alpaca setup start
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream

API_KEY = "PKEA6SOXUOGIZY6PQ8J7"
API_SECRET = "EJe9TzECKGji8uPGySTyc3JJDCuJ5g7gx8UczOO7"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

alpaca = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, api_version='v2')
# end alpaca setup
logger = logging.getLogger("log"+str(datetime.datetime.now()))
# Configure logger to write to a file...

def my_handler(type, value, tb):
    logger.exception("Uncaught exception: {0}".format(str(value)))

# Install exception handler
sys.excepthook = my_handler

# prices = np.array(fd["Close"])
# sma = talib.SMA(prices, timeperiod=25)
# ams = np.roll(sma, -17)
# dates = np.array(fd["date"])
# plt.plot(ams, zorder=5)
# plt.plot(prices, zorder=0)
hist = []
count = 0
actlist = []
global actionlist
global finalactionlist
global parabolas
global plottedparabolas
global goodparabolas
global ams
global df
global dateline
global fd
global dayProfits

finalactionlist = {}
plottedparabolas = {}
parabolas = {}
goodparabolas = {}


def findDateLine(date, stock):
    print(date + stock)
    # x= re.search(r'{0}:\w\w\w'.format(date))
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
    print("Vertex: (", (-b / (2 * a)), ", "
          , (((4 * a * c) - (b * b)) / (4 * a)), ")")

    print("Focus: (", (-b / (2 * a)), ", "
          , (((4 * a * c) - (b * b) + 1) / (4 * a)), ")")

    print("Directrix: y="
          , (int)(c - ((b * b) + 1) * 4 * a))

    return vertex


global NewVal
NewVal = False


def filterParabolaSignals():
    global finalactionlist
    global NewVal
    print("filtering parabolas")
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
            finalactionlist[val] = "BUY"
            print("NEW VALUE")
            NewVal = True
        else:
            if negthreshold > parabolas[val][0] > negthresholdmax:
                print("good down")
                goodparabolas[val] = parabolas[val]
                finalactionlist[val] = "SELL"
                print("NEW VALUE")
                NewVal = True
            else:
                print(str(val) + " not within threshhold removing")
                # parabolas.pop(val)


def decidesignals2(signals):
    global parabolas

    print(signals)
    for index, val in enumerate(signals):

        print(index)
        print(val)
        print("parabolas pressent")
        # print(parabolas)
        if val in parabolas or val < 20:
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
        print(len(y))
        print(x)
        print(y)
        plt.scatter(x[0], y[0])
        plt.scatter(x[len(x) - 1], y[len(y) - 1])
        print("#################################")
        fit_params, pcov = scipy.optimize.curve_fit(parabola, x, y)
        parabolas[val] = fit_params

    print(parabolas)
    return parabolas


def decidesignals(signals):
    global finalactionlist
    print(signals)
    for index, val in enumerate(signals):
        print("deciding signals")
        print(val)
        x = np.linspace(val - 10, val + 10, val + 11)
        y_with_errors = fd['Close'][val - 10:]
        y_end_with_errors = y_with_errors[:-1 * (len(fd['Close']) - val - 1 - 10)]
        print("y end ")
        print(y_end_with_errors)
        print("y with errors")
        print(y_with_errors)
        # np.linspace(start, stop, num)

        # fit_params, pcov = scipy.optimize.curve_fit(parabola, x, y_with_errors)

    #     for x in range(val + 5, val + 10):
    #         price = ams[x]
    #         # print(x)
    #         # print("deciding signals")
    #         # print(sig)
    #         # print(price)
    #         actionlist = []
    #         if val < price:
    #             print("buy")
    #             actionlist.append(True)
    #         else:
    #             if val > price:
    #                 print("sell")
    #                 actionlist.append(False)
    #         falsecount = 0
    #         truecount = 0
    #         for act in actionlist:
    #             if act is False:
    #                 falsecount = + 1
    #             if act is True:
    #                 truecount = + 1
    #     if falsecount > truecount:
    #         print(val)
    #         finalactionlist[val] = False
    #     if truecount > falsecount:
    #         finalactionlist[val] = True
    # print(finalactionlist)


# initialize dataset


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

    # print(str(maxs[0]))
    maxs1 = str(maxs[0])[2:][:-1].split(" ")
    # maxs1.remove('')
    for index, val in enumerate(maxs1):
        if "\n" in val:
            print(val)
            val = val.replace('\n', "")
            maxs1[index] = val
    # print(maxs1)
    # print(maxs[0][0])
    maxs11 = filtersignals(maxs1)
    maxsnp = np.array(maxs11)
    maxs11 = dropna(maxsnp)
    # maxs11 = maxs11[np.logical_not(np.isnan(maxs11))]
    maxs2 = []
    for x in maxs11:
        maxs2.append(int(x))
    # print(maxs2)
    # decidesignals(maxs2)
    decidesignals2(maxs2)
    maxs2 = np.array(maxs2)
    # print(maxs)
    ymaxs = []
    # print(np.size(maxs2))
    for x in maxs2:
        print(int(x))
        ymaxs.append(ams[x])
    # axsnp = np.array(ymaxs)
    print(ymaxs)
    filter(lambda v: v == v, ymaxs)

    print(ymaxs)
    return maxs2, ymaxs


def calcProfits():
    previoussignal = ""
    totalchange = 0
    previousval = 0
    for val in finalactionlist:
        print(val)
        print(cumulativeprices[val])
        tradeprice = cumulativeprices[val]
        difference = tradeprice - cumulativeprices[previousval]
        totalchange += difference
        print(f"difference between {previousval} and {val} = {difference} ")
        print(f"difference between {tradeprice} and {cumulativeprices[previousval]} = {difference} ")

        print(
            f"difference between {finalactionlist[previousval] if previousval != 0 else 'None'} and {finalactionlist[val] if val != 0 else 'None'} = {difference} ")
        print(difference)
        print("totalchange")
        print(totalchange)
        previousval = val
    print("################################################")
    print("Total Change: " + str(totalchange))


global orders
global totalQty
global totalProfit
totalProfit = 0
orders = {}
totalQty = 0


def historybroker(action, qty):
    global orders, difference
    global totalQty
    global totalProfit
    difference = 0
    print("action recieved: " + action)
    currentPrice = cumulativeprices[-1]
    if "BUY" in action:
        print("Buy order Created with: " + str(qty) + " qty, ")

        if totalQty == 0:
            orders[currentPrice] = "BUY"
            difference = 0
            totalQty = qty
            print("totalQty: " + str(totalQty))
        else:
            print("ALready In positon Liquidating")
            prevOrder = list(orders.keys())[-1]
            if orders[prevOrder] is not None:
                # prevOrder = orders[-1]
                prevOrderPrice = prevOrder
                if totalQty < 0:
                    difference = prevOrderPrice - currentPrice
                    print("difference: " + str(difference))
                    orders[currentPrice] = "BUY"
                    totalQty = 0

    if "SELL" in action:
        print("Sell order Created with: " + str(qty) + " qty")

        if totalQty == 0:
            orders[currentPrice] = "SELL"
            difference = 0
            totalQty = qty
            print("totalQty: " + str(totalQty))
        else:
            print("ALready In positon Liquidating")
            prevOrder = list(orders.keys())[-1]
            if orders[prevOrder] is not None:
                # prevOrder = orders[-1]
                prevOrderPrice = prevOrder
                if totalQty < 0:
                    difference = currentPrice - prevOrderPrice
                    print("difference: " + str(difference))
                    orders[currentPrice] = "SELL"
                    totalQty = 0

    totalProfit = totalProfit + difference
    print("total profit: " + str(totalProfit))
    print("totalQty: " + str(totalQty))
    print("orders")
    print(orders)


class livebroker:

    def __init__(self):
        global orders, difference
        global totalQty
        global totalProfit
        difference = 0
        # print("action recieved: " + action)
#        currentPrice = cumulativeprices[-1]
        self.totalQty = totalQty
        self.orders = orders
        self.difference = difference
        self.totalProfit = totalProfit

    def awaitMarketOpen(self):
        global prediction
        try:
            isOpen = alpaca.get_clock().is_open
        except:
            time.sleep(30)
            isOpen = alpaca.get_clock().is_open
        print("isOpen: " + str(isOpen))
        while not isOpen:
            clock = alpaca.get_clock()
            openingTime = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            timeToOpen = int((openingTime - currTime) / 60)
            print(str(timeToOpen) + " minutes til market open.")
            time.sleep(60)
            if (timeToOpen <= 10):
                print("10 Minutes till opening")

            try:
                isOpen = alpaca.get_clock().is_open
            except:
                print("error in awaitMarketOpen")

    def closingTime(self):
        clock = alpaca.get_clock()
        closingTime = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
        currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
        self.timeToClose = closingTime - currTime

        if (self.timeToClose < (60 * 15)):
            # Close all positions when 15 minutes til market close.
            print("Market closing soon.")

            return True
        else:
            return False

    def awaitMarketClose(self):
        while self.timeToClose > (60 * 15):
            try:
                clock = alpaca.get_clock()
            except:
                time.sleep(30)
                clock = alpaca.get_clock()
            closingTime = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            self.timeToClose = closingTime - currTime
            time.sleep(30)
            if self.timeToClose < (60 * 15):
                # Close all positions when 15 minutes til market close.
                print("Market closing soon.  Closing positions.")
                alpaca.close_all_positions()
                # side = 'sell'

    def isTradable(self, symbol):
        try:
            trad = alpaca.get_asset(symbol)
            if trad.tradable:
                return True
            else:
                return False
        except:
            return False

    def getbuying(self):
        acount = alpaca.get_account()
        buyingpower = acount.buying_power
        print("Buying Power: " + str(buyingpower))
        if buyingpower == 0:
            print("0 is it, " + buyingpower)
            return 1000
        else:
            buyingminus = float(buyingpower) * 0.005
            return float(buyingpower) - float(buyingminus)

    def getqty(self):
        try:
            orders = alpaca.list_orders(status='open')
            print(orders)
            qty = orders[0].qty
            print(qty)
        except:
            qty = self.getbuying()/alpaca.get_latest_bar("AAPL").c
            qty = qty*0.9
        return qty
    def profit(self):
        account = alpaca.get_account()
        balance_change = float(account.equity) - float(account.last_equity)
        print(f'Today\'s portfolio balance change: ${balance_change}')
        return round(balance_change, 2)

    def buy(self, qty):
        print("Buy order Created with: " + str(qty) + " qty, ")
        currentPrice = cumulativeprices[-1]
        print("currunt price: " + str(currentPrice))
        if self.totalQty == 0:
            self.orders[currentPrice] = "BUY"
            difference = 0
            self.totalQty = qty
            print("totalQty: " + str(self.totalQty))

            alpaca.submit_order(
                symbol="IBM",
                qty=self.totalQty,
                side='buy',
                type='market',
                time_in_force='day',
                order_class='oto',
                stop_loss={'stop_price': currentPrice * 0.95}
            )


        else:
            print("ALready In positon Liquidating")
            prevOrder = list(orders.keys())[-1]
            if self.orders[prevOrder] is not None:
                # prevOrder = orders[-1]
                prevOrderPrice = prevOrder
                if self.totalQty < 0:
                    difference = prevOrderPrice - currentPrice
                    print("difference: " + str(difference))
                    self.orders[currentPrice] = "BUY"
                    self.totalQty = self.totalQty + qty
                    print("totalQty: " + str(self.totalQty))

                    alpaca.submit_order(
                        symbol="IBM",
                        qty=self.totalQty,
                        side='buy',
                        type='market',
                        time_in_force='day',
                        order_class='oto',
                        stop_loss={'stop_price': currentPrice * 0.95}
                    )

    def sell(self, qty):
        print("SELL order Created with: " + str(qty) + " qty, ")
        currentPrice = cumulativeprices[-1]
        print("currunt price: " + str(currentPrice))
        if self.totalQty == 0:
            self.orders[currentPrice] = "SELL"
            difference = 0
            self.totalQty = -1 * qty
            print("totalQty: " + str(self.totalQty))

            alpaca.submit_order(
                symbol="IBM",
                qty=abs(self.totalQty),
                side='sell',
                type='market',
                time_in_force='day',
                order_class='oto',
                stop_loss={'stop_price': currentPrice * 1.05}
            )


        else:
            print("ALready In positon Liquidating")
            prevOrder = list(orders.keys())[-1]
            if self.orders[prevOrder] is not None:
                # prevOrder = orders[-1]
                prevOrderPrice = prevOrder
                if self.totalQty > 0:
                    difference = prevOrderPrice - currentPrice
                    print("difference: " + str(difference))
                    self.orders[currentPrice] = "SELL"
                    self.totalQty = -1 * (self.totalQty + qty)
                    print("totalQty: " + str(self.totalQty))

                    alpaca.submit_order(
                        symbol="IBM",
                        qty=abs(self.totalQty),
                        side='sell',
                        type='market',
                        time_in_force='day',
                        order_class='oto',
                        stop_loss={'stop_price': currentPrice * 1.05}
                    )

    def reallivebroker(self, action, qty):
        if "BUY" in action:
            print("Buy order Created with: " + str(qty) + " qty, ")
            self.buy(self.getqty())

        if "SELL" in action:
            print("Sell order Created with: " + str(qty) + " qty")
            self.sell(self.getqty())
    # totalProfit = totalProfit  + difference
    # print("total profit: " + str(totalProfit))
    # print("totalQty: " + str(totalQty))
    print("orders")
    print(orders)


def getCurrentPosition():
    if totalQty == 0:
        return "NONE"
    if totalQty > 0:
        return "BUY"
    if totalQty < 0:
        return "SELL"


cumulativeprices = []
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.use('TkAgg')
fig, ax = plt.subplots()
########
plt.legend(loc="upper left")
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 5))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

BUYQTY = 10


def onTick(index, prices, livedata):
    global ams
    global totalProfit
    global NewVal
    global dayProfits
    print("new TIck")
    # print(index)
    index = index
    # row1 = fd['Close']
    # print(row1)
    prices = prices
    # print(prices)
    if livedata is not None:
        broker = livebroker().reallivebroker
    else:
        broker = historybroker
    if len(cumulativeprices) > 5:
        sma = talib.SMA(np.array(cumulativeprices), timeperiod=38)
        ema = talib.EMA(np.array(cumulativeprices), timeperiod=38)
        testma = talib.SMA(np.array(cumulativeprices), timeperiod=15)
        ams = np.roll(sma, -22)
        plt.plot(ams, color="blue", label="AMS")
        plt.plot(ema, color="orange", label="EMA")
        # plt.plot(testma, color="red", label="TESTMA")

        xmaxs, ymaxs = getSignals()
        plt.scatter(xmaxs, ymaxs)
        # print(parabolas)
        if len(parabolas) >= 1:
            print("plotting parobalsf")
            # print(parabolas)
            # print(goodparabolas)
            filterParabolaSignals()

            if NewVal is True:
                position = getCurrentPosition()
                print("BUYING or SELLING")
                NewVal = False
                val = list(finalactionlist.keys())[-1]
                action = finalactionlist[val]
                print(val)
                print("we actually want to " + action)
                if action in "BUY":
                    if position not in "BUY":
                        print("BUY order")
                        livebroker().reallivebroker(action="BUY", qty=BUYQTY)
                        print("totalQty: IN MAIN" + str(totalQty))
                    else:
                        print("already in " + position + " position we want to BUY")
                        print("orders")
                        print(orders)

                if action in "SELL":
                    print("SELL order")
                    if position not in "SELL":
                        livebroker().reallivebroker(action="SELL", qty=BUYQTY)
                        print("totalQty: IN MAIN" + str(totalQty))
                    else:
                        print("already in " + position + " position we want to SELL")
                        print("orders")
                        print(orders)

                NewVal = False

                print("finalactuionlist")
                print(finalactionlist)
                print("totalProfit: IN BUY " + str(totalProfit))

            for index, val in enumerate(goodparabolas):
                print(val)
                print(plottedparabolas.keys())

                # d = list(val)

                fit_params = parabolas[val]
                # fit_params = val[]
                parvertext = parabolavertex(*fit_params)
                print(parvertext)
                if goodparabolas[val][0] < 0:
                    print("sell")
                    plt.annotate("SELL PV", parvertext)
                if goodparabolas[val][0] > 0:
                    print("BUY")
                    plt.annotate("BUY PV", parvertext)

                plt.scatter(parvertext[0], parvertext[1], label="PV")
                x = np.linspace(val - 25, val + 5, 31)
                y_fit = parabola(np.array(x), *fit_params)

                plottedparabolas[val] = [x, y_fit]
                print(str(val) + " plotted val" + str(x), str(y_fit))
                # print(plottedparabolas)
                plt.plot(x, y_fit, color="magenta", zorder=10, linewidth=3.0)

                # print("NewVal: " + str(NewVal))

                if val in plottedparabolas.keys():
                    print("val bought already")
                    # plt.plot(plottedparabolas[val][0], plottedparabolas[val][1])
                    continue

                else:
                    print("val not bought already")

        for index, val in enumerate(xmaxs):
            print(val)
            try:
                print(xmaxs)
                # print(finalactionlist)
                # print(finalactionlist[val])
                plt.annotate("Buy" if finalactionlist[val] is True else "Sell", (val, ymaxs[index]))
            except:
                continue

    cumulativeprices.append(prices)
    # print("price list")
    # print(cumulativeprices)
    plt.plot(cumulativeprices)
    plt.draw()
    print("index")
    print(index)
#    print(len(fd) + 3000)
    #length = len(fd) + 3000
    # if index <= 15 and index >= 10:
    #     print("done waiting")
    #     # calcProfits()
    #     print("TOTAL PROFITS: " + str(totalProfit))
    #     time.sleep(1)
    #     dayProfits.append(totalProfit)

    plt.pause(0.001)

    plt.clf()


def runBackTesting(startDate, numDays):
    global dayProfits
    dayProfits = []
    # sDate = datetime(startDate[])
    aDate = startDate.split("-")
    print(aDate)
    sDate = datetime.datetime(int(aDate[0]), int(aDate[1]), int(aDate[2]))
    for x in range(0, numDays):

        print("dayprofits")
        print(dayProfits)
        global df
        global dateline
        global fd
        stock = "IBM"
        date = sDate + datetime.timedelta(days=x)
        date_string = date.strftime("%Y-%m-%d")
        date = date_string
        # load dataset

        df = pandas.read_csv("./datasets/" + stock + ".csv")
        print(df)
        dateline = findDateLine(date, stock)
        print("datline: {0}".format(dateline))
        fd = df.iloc[dateline:]
        print(fd)

        for index, row in enumerate(fd.iterrows()):
            onTick(index, fd['Close'][index + dateline])


def runLiveTrading():
    global dayProfits
    dayProfits = []
    stock = "AAPL"

    async def trade_callback(t):
        print('trade', t)

    async def quote_callback(q):
        print('quote', q)
        onTick(datetime.datetime.fromtimestamp(q.timestamp // 1000000000), q.close, q)

    # Initiate Class Instance
    stream = Stream(API_KEY,
                    API_SECRET,
                    base_url=APCA_API_BASE_URL,
                    data_feed='iex')  # <- replace to SIP if you have PRO subscription

    # subscribing to event
    #stream.subscribe_quotes()
    stream.subscribe_bars(quote_callback, stock)
    # stream.subscribe_quotes(quote_callback, 'IBM')
    livebroker().awaitMarketOpen()
    stream.run()
    livebroker().awaitMarketClose()
    stream.stop_ws()
    # wait for market to open before trading

def testFunctions():
    livebroker().reallivebroker(action="SELL", qty=0)
    livebroker().reallivebroker(action="BUY", qty=0)
# runBackTesting("2020-12-29", 2)
runLiveTrading()
#testFunctions()
# print(dayProfits)
# for local minima
# argrelextrema(ams, np.less)
print(actlist)
# plt.xticks(dates)
# plt.show()
# print(maxs2)
print(finalactionlist)
