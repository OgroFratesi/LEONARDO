import pandas as pd
import numpy as np
from napoleon_realtime import *
import boto3
from config import *



names_crypto = ['SOLUSDT','ADAUSDT','XRPUSDT','MATICUSDT','DOGEUSDT','FTMUSDT','LUNAUSDT']
indicators = {'EMA5':[15,30,45],'low5':[5,30], 'diff%':[10], 'rsi5':[14], 'BB5':[10,20],'BB1':[20], 'rsi1':[14]}

best_combi_xrp = pd.read_csv('data/combinations_XRPUSDT_jun2023.csv',index_col=0)
best_combi_sol = pd.read_csv('data/combinations_SOLUSDT_jun2023.csv',index_col=0)
best_combi = pd.concat([best_combi_xrp,best_combi_sol])
target_vector = best_combi.values[:,:-4]

trader = LEONARDO(indicators, names_crypto, target_vector, trade_size=200, 
                 API_KEY=API_KEY,
                 SECRET_KEY=SECRET_KEY)


if __name__ == "__main__":

    time.sleep(3)

    print('LEONARDO ha comenzado. Buena suerte.')
    
    trader.run()