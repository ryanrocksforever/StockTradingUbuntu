import pandas
import talib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from talib import MA_Type

df = pandas.read_csv("./datasets/IBM.csv")
print(df)
fd = df.iloc[3010:]
print(fd)
prices = np.array(fd["Close"])
sma = talib.SMA(prices, timeperiod=25)
ams = np.roll(sma, -17)
dates = np.array(fd["date"])
plt.plot(ams, zorder=5)
plt.plot(prices, zorder=0)
hist = []
count = 0
actlist = []
global actionlist
global finalactionlist
finalactionlist = {}


def filtersignals(signals):
    global actionlist
    # signals = signals[0]
    #print(type(signals))
    prev = 0
    while('' in signals) :
        signals.remove('')
    for index, sig in enumerate(signals):
        print(str(sig) + ", " + str(prev))
        #print(index)
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
    #print("final actrion list")
    #print(finalactionlist)
    return signals

def decidesignals(signals):
    global finalactionlist
    print(signals)
    for index, val in enumerate(signals):
        print("deciding signals")
        print(val)
        for x in range(val + 5, val + 10):
            price = ams[x]
            #print(x)
            #print("deciding signals")
            #print(sig)
            #print(price)
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
            for val in actionlist:
                if val is False:
                    falsecount =+ 1
                if val is True:
                    truecount =+ 1
        if falsecount > truecount:
            print(val)
            finalactionlist[val] = False
        if truecount > falsecount:
            finalactionlist[val]= True
    print(finalactionlist)
# for local maxima
maxs = argrelextrema(ams, np.greater)

print(str(maxs[0]))
maxs1 = str(maxs[0])[2:][:-1].split(" ")
maxs1.remove('')
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
maxs2 = np.array(maxs2)
print(maxs)
ymaxs = []
print(np.size(maxs2))
for x in maxs2:
    print(int(x))
    ymaxs.append(ams[x])

print(ymaxs)

plt.scatter(maxs2, ymaxs, zorder=10)




for index, val in enumerate(maxs2):
    print(val)
    plt.annotate("signal", (val, ymaxs[index]))

# for local minima
# argrelextrema(ams, np.less)
print(actlist)
# plt.xticks(dates)
plt.show()
print(maxs2)
print(finalactionlist)
