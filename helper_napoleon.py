import pandas as pd
import time
import boto3
import itertools
import ta
from datetime import datetime
import warnings
from tqdm import tqdm
import numpy as np


def convert_df(historical):

    time = []
    open = []
    high = []
    low = []
    close = []
    volume = []

    for e in range(len(historical)):
        hist_data = historical[e]
        time.append(convert_time(hist_data[0]))
        open.append(hist_data[1])
        high.append(hist_data[2])
        low.append(hist_data[3])
        close.append(hist_data[4])
        volume.append(hist_data[5])

    df = pd.DataFrame({'time':time,'open':open, 'high':high, 'low':low, 'close':close, 'volume':volume})
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    return df

def convert_time(time):
    d = datetime.fromtimestamp(int(str(time))/1000)
    return d



def create_indicators(indicators,df_use, timeframe):


    for indicator, values in indicators.items():
        if timeframe in indicator:
            if (indicator == 'EMA5')|(indicator == 'EMA1'):
                nums = indicators[indicator]
                pairs = list(itertools.combinations(nums, 2))
                # Remove duplicates where 5,10 is same as 10,5
                pairs = [(a,b) if a<b else (b,a) for a,b in pairs]
                pairs = list(set(pairs))
                for v in nums:
                    if v == 15:
                        df_use[f'close_{indicator}_{v}'] = np.where(df_use.close > df_use.close.ewm(span=v, adjust=False).mean(), 1, -1)
                for v in pairs:
                    df_use[f'{indicator}_{v}'] = np.where(df_use.close.ewm(span=v[0], adjust=False).mean() > df_use.close.ewm(span=v[1], adjust=False).mean(), 1, -1)
            elif (indicator == 'low5')|(indicator == 'low1'):
                df_use[f'{indicator}'] = np.where(df_use.close.rolling(values[0]).min() > df_use.close.rolling(values[1]).min(), 1, -1)
            elif indicator == 'high5':
                df_use[f'{indicator}'] = np.where(df_use.close.rolling(values[0]).max() < df_use.close.rolling(values[1]).max(), 1, -1)
            elif (indicator == 'rsi5')|(indicator == 'rsi1'):
                for v in values:
                    rsi = ta.momentum.rsi(close=df_use.close, window=v, fillna=True)
                    df_use[f'{indicator}_{v}'] = np.where(((rsi) < 30 & (rsi > 70)), 1, -1)
            elif (indicator == 'adx5')|(indicator == 'adx1'):
                for v in values:
                    adx = ta.trend.ADXIndicator(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use[f'{indicator}_{v}'] = np.where((adx.adx() > 30), 1, -1)
            elif indicator == '5d+-':
                for v in values:
                    adx = ta.trend.ADXIndicator(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use[f'{indicator}_{v}'] = np.where((adx.adx_pos() > adx.adx_neg()), 1, -1)
            elif (indicator == 'volatility5')|((indicator == 'volatility1')):
                for v in values:
                    volatility = ta.volatility.AverageTrueRange(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use[f'{indicator}_{v}'] = np.where(((volatility.average_true_range() / df_use.close * 100) > 0.8), 1, -1)
            elif indicator == 'diff%':
                for v in values:
                    df_use[f'{indicator}_{v}'] = np.where(((df_use.high.rolling(v).max() - df_use.close) / df_use.high.rolling(v).max() * 100) > 2, 1, -1)
            elif (indicator == 'BB5')|(indicator == 'BB1'):
                for v in values:
                    indicator_bb = ta.volatility.BollingerBands(close=df_use["close"], window=v, window_dev=2)
                    df_use[f'{indicator}_{v}'] = np.where((indicator_bb.bollinger_hband_indicator() > 0) | (indicator_bb.bollinger_lband_indicator() > 0), 1, -1)

    return df_use


def create_combinations(indicators, df_all):

    for timeframe in ['1hour','5minutes']:
        if timeframe == '1hour':
            df_use = df_all[df_all.timeframe == '1hour']
            df_hour = create_indicators(indicators, df_use, '1')
        elif timeframe == '5minutes':
            df_use = df_all[df_all.timeframe == '5minutes']
            df_5minute = create_indicators(indicators, df_use, '5')

    original_columns = ['time','open', 'high','low','close','volume','timeframe','adj_close']
    columnas_1 = [col for col in df_hour.columns if col not in original_columns]
    columnas_1.append('time1')

    df_hour['time'] = pd.to_datetime(df_hour['time'])
    df_hour['time1'] = df_hour['time'].dt.tz_localize(None)
    df_hour['time1'] = df_hour.time1.shift(-1)
    df_hour.drop('time',axis=1,inplace=True)

    df_hour.reset_index(inplace=True, drop=True)

    df_5minute.reset_index(inplace=True,drop=True)
    df_5minute['time'] = pd.to_datetime(df_5minute['time'])
    df_5minute['time1'] = df_5minute['time'].apply(lambda x: x.strftime("%Y-%m-%d %H"))
    df_5minute['time1'] = df_5minute['time1'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H'))

    df_use = df_5minute.merge(df_hour[columnas_1], on=['time1','crypto'], how='left')

    return df_use


def check_strategy(uniques, df_use):

    original_columns = ['crypto','time', 'open', 'high','low','close','volume','time1', 'adj_close', 'timeframe']
    columnas = len([col for col in df_use.columns if col not in original_columns])
    columnas_name = [col for col in df_use.columns if col not in original_columns]

    for unique_combi in uniques:
        df_array = df_use.loc[:, columnas_name].values
      
        df_use_calculate = df_use.copy()
        zero_indices = [i for i in range(len(unique_combi)) if unique_combi[i] == 0]

        doty = np.dot(df_array ,unique_combi.reshape(columnas, -1))
        equal_to = columnas - len(zero_indices)
        df_use_calculate['target'] = np.where(doty == equal_to, 1, 0)

        if df_use_calculate.shape[0] < 3:
            return pd.DataFrame()
        if df_use_calculate.iloc[-2, -1] == 1:
            print(df_use_calculate.crypto.values[0])
            return df_use_calculate


    return df_use_calculate

