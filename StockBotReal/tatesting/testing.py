from alpaca_trade_api.stream import Stream
import alpaca_trade_api as tradeapi
from datetime import datetime

timestamp = 1545730073
dt_object = datetime.fromtimestamp(timestamp)

print("dt_object =", dt_object)
print("type(dt_object) =", type(dt_object))


async def trade_callback(t):
    print('trade', t)


async def quote_callback(q):
    print('quote', q)
    #print(q[0]['timestamp'])
    timestamp = q.timestamp
    print(repr(str(timestamp.to_datetime64())))
    object_methods = [method_name for method_name in dir(timestamp)
                      if callable(getattr(timestamp, method_name))]
    print("object Methods: ", object_methods)
    #dt_object = datetime.fromtimestamp(timestamp)

    #print("dt_object =", dt_object)

    print("askPrice: ", q.ask_price)

async def bar_callback(q):
    print('quote', q)
    #print(q[0]['timestamp'])
    timestamp = q.timestamp
    print(repr(str(timestamp.to_datetime64())))
    object_methods = [method_name for method_name in dir(timestamp)
                      if callable(getattr(timestamp, method_name))]
    print("object Methods: ", object_methods)
    #dt_object = datetime.fromtimestamp(timestamp)

    #print("dt_object =", dt_object)

    print("askPrice: ", q.ask_price)


API_KEY = "PKEA6SOXUOGIZY6PQ8J7"
API_SECRET = "EJe9TzECKGji8uPGySTyc3JJDCuJ5g7gx8UczOO7"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
# Initiate Class Instance
stream = Stream(API_KEY,
API_SECRET,
 base_url=APCA_API_BASE_URL,
          data_feed='iex')  # <- replace to SIP if you have PRO subscription
alpaca = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, api_version='v2')
# subscribing to event
#stream.subscribe_trades(trade_callback, 'AAPL')
#stream.subscribe_bars(bar_callback, 'IBM')

#stream.run()

def getqty():
    try:
        orders = alpaca.list_orders(status='open')
        print(orders)
        qty = orders[0].qty
        print(qty)
    except:
        print(alpaca.get_latest_bar('AAPL').c)
        qty = 100000/alpaca.get_latest_bar('AAPL').c
    return qty

x = getqty()
print(x)