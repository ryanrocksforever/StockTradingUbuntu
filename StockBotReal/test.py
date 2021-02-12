import bulbea as bb
import os
import get_dataset
symbol = "SPY"
os.environ["BULBEA_QUANDL_API_KEY"] = 'YNtcRm5xsr1L4gUYbtFi'
share = bb.Share('WIKI', symbol, database="alpha")
#print(share.data)
# for col in share.data.columns:
#     print(col)
#     #print(share.data[col])
share.data = get_dataset.get_historical_data(symbol)
#print(share.data[["Adj. Low"]])
#for columnName, columnData in share.data[["Adj. Low"]].iteritems():
#    print(columnData["2004-08-19"])
from bulbea.learn.evaluation import split
Xtrain, Xtest, ytrain, ytest = split(share, 'Close', normalize = True)
print(share.data['Close'])
import numpy as np
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
Xtest  = np.reshape(Xtest,  (Xtest.shape[0],  Xtest.shape[1], 1))
print(Xtrain.shape)
print(Xtest.shape)
testtt = np.array_equal(Xtrain, Xtest)
print(testtt)

from bulbea.learn.models import RNN
#from bulbea.learn.models.ann import RNNCell
rnn = RNN([1, 100, 100, 1]) # number of neurons in each layer
rnn.fit(Xtrain, ytrain)
from sklearn.metrics import mean_squared_error
#print(Xtest)
p = rnn.predict(Xtest)
#p = rnn.predict(p)
m = mean_squared_error(ytest, p)
print(m)
#print(ytest)
#0.00042927869370525931
import matplotlib.pyplot as pplt
import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color='b', label='ytest: DataSet')
pplt.plot(ytest, color='b')

print(p[0])
counter = 2
prices = []
for i in p:

    row = share.data.iloc[[counter]]
    #print(row['Close'])
    close = row['Close']
    if i > 0:
        i = i +1
    else:
        if i <0:
            i = i -1

    prices.append(close * i)
    counter += 1
print("prices")
print(prices)
red_patch1 = mpatches.Patch(color='m', label='p:Prediction')
pplt.plot(p, color='m')
pplt.legend(handles=[red_patch, red_patch1])
pplt.show()