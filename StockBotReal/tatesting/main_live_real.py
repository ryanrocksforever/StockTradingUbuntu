import time

import pandas
import talib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import argrelextrema
from talib import MA_Type
import ta

df = pandas.read_csv("./datasets/IBM.csv")
print(df)
fd = df.iloc[3010:]
print(fd)
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
finalactionlist = {}


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


def decidesignals2(signals):
    global parabolas
    parabolas = {}
    print(signals)
    for index, val in enumerate(signals):
        print(index)
        print(val)
        if val <= 6:
            continue
        x = np.linspace(val - 50-36, val + 5, 93)

        print(x)
        print("#################################")

        y = fd['Close'][val - 50:]
        y = y[:-1*(len(y)-val-5)]
        print(y)
        y = y.tolist()

        print(len(y))
        print(y)
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

        for x in range(val + 5, val + 10):
            price = ams[x]
            # print(x)
            # print("deciding signals")
            # print(sig)
            # print(price)
            actionlist = []
            if val < price:
                print("buy")
                actionlist.append(True)
            else:
                if val > price:
                    print("sell")
                    actionlist.append(False)
            falsecount = 0
            truecount = 0
            for act in actionlist:
                if act is False:
                    falsecount = + 1
                if act is True:
                    truecount = + 1
        if falsecount > truecount:
            print(val)
            finalactionlist[val] = False
        if truecount > falsecount:
            finalactionlist[val] = True
    print(finalactionlist)


shortPeriod = 14
longPeriod = 200


def SMASig(close, sPeriod, lPeriod):
    shortSMA = ta.SMA(close, sPeriod)
    longSMA = ta.SMA(close, lPeriod)
    smaSell = ((shortSMA <= longSMA) & (shortSMA.shift(1) >= longSMA.shift(1)))
    smaBuy = ((shortSMA >= longSMA) & (shortSMA.shift(1) <= longSMA.shift(1)))
    return smaSell, smaBuy, shortSMA, longSMA


def getSignals():
    maxs = argrelextrema(ams, np.greater)

    print(str(maxs[0]))
    maxs1 = str(maxs[0])[2:][:-1].split(" ")
    # maxs1.remove('')
    for index, val in enumerate(maxs1):
        if "\n" in val:
            print(val)
            val = val.replace('\n', "")
            maxs1[index] = val
    print(maxs1)
    # print(maxs[0][0])
    maxs11 = filtersignals(maxs1)
    maxs2 = []
    for x in maxs11:
        maxs2.append(int(x))
    print(maxs2)
    decidesignals(maxs2)
    decidesignals2(maxs2)
    maxs2 = np.array(maxs2)
    print(maxs)
    ymaxs = []
    print(np.size(maxs2))
    for x in maxs2:
        print(int(x))
        ymaxs.append(ams[x])

    print(ymaxs)
    return maxs2, ymaxs


cumulativeprices = []
import matplotlib

matplotlib.use('TkAgg')

########
for index, row in enumerate(fd.iterrows()):
    # print(index)
    index = index + 3010
    row1 = fd['Close']
    # print(row1)
    prices = row1[index]
    print(prices)
    if len(cumulativeprices) > 5:
        sma = talib.SMA(np.array(cumulativeprices), timeperiod=30)
        ams = np.roll(sma, -18)
        plt.plot(ams)
        xmaxs, ymaxs = getSignals()
        plt.scatter(xmaxs, ymaxs)
        print(parabolas)
        if len(parabolas) >= 1:
            print("plotting [parobalsf")
            for index, val in enumerate(parabolas):
                print (val)
                fit_params = val
                x = np.linspace(index-50, index+5, 56)
                y_fit = parabola(np.array(x), *fit_params)
                plt.plot(x, y_fit)
        for index, val in enumerate(xmaxs):
            print(val)
            try:
                print(xmaxs)
                print(finalactionlist)
                print(finalactionlist[val])
                plt.annotate("Buy" if finalactionlist[val] is True else "Sell", (val, ymaxs[index]))
            except:
                continue


    cumulativeprices.append(prices)
    # print("price list")
    # print(cumulativeprices)
    plt.plot(cumulativeprices)
    plt.draw()
    print(index)
    print(len(fd) + 3000)
    if index >= (len(fd) + 3000) - 1:
        time.sleep(60)
    plt.pause(0.001)
    plt.clf()
#######

# for local minima
# argrelextrema(ams, np.less)
print(actlist)
# plt.xticks(dates)
# plt.show()
# print(maxs2)
print(finalactionlist)
