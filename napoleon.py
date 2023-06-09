import pandas as pd
import numpy as np
from napoleon_realtime import *


names_crypto = ['BTCUSDT','SOLUSDT','ADAUSDT','XRPUSDT','MATICUSDT','ALPHAUSDT','OGUSDT','DOGEUSDT','FTMUSDT','LUNAUSDT',
                'MTLUSDT', 'RADUSDT', 'TOMOUSDT', 'MDTUSDT']
indicators = {'EMA5':[15,30,45],'low5':[5,30], 'diff%':[10], 'rsi5':[14], 'BB5':[10,20],'BB1':[20], 'rsi1':[14]}

best_combi = pd.read_csv('binance/data/best_combi.csv',index_col=0)
target_vector = best_combi.values[:,:-3]

trader = LEONARDO(indicators, names_crypto, target_vector, trade_size=200)


if __name__ == "__main__":
    
    trader.run()