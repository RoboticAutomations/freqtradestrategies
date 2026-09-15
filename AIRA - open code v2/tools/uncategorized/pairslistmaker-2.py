import ccxt
import time
import json

n = 0
ex = ccxt.bitrue({'verbose': False}) # log HTTP requests
ex.load_markets() # request markets

markets = list(ex.markets.keys())

### exclude leverage markets
for key in markets:
    if key.__contains__("USDT") and not key.__contains__("3"):
        n = n + 1
        print('    "{}",'.format(key))
print(f'Number of markets: {n}')

