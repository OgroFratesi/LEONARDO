from time import sleep
import pandas as pd
from datetime import datetime, timedelta
# import binance.config as config
import itertools
import numpy as np
from datetime import datetime 
import ta
import decimal
import tqdm



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


def convert_15(df_use):

    df_use['time'] = pd.to_datetime(df_use['time'])
    df_use['timeh'] = df_use['time'].apply(lambda x: x.strftime("%H-%M"))
    df_use['timeh'] = df_use['timeh'].apply(lambda x: datetime.strptime(str(x), '%H-%M').minute)

    def sum_minutes(x):
        if x in (5,20,35,50):
            return 5
        elif x in (10, 25, 40, 55):
            return 10
        else:
            return 0

    df_use['add'] = df_use.timeh.apply(sum_minutes)

    df_use['15m'] = df_use.apply(lambda row: row['time'] - timedelta(minutes=row['add']), axis=1 )

    return df_use


def convert_4(df_use, horas4):

        df_use['time'] = pd.to_datetime(df_use['time'])
        df_use['time4'] = df_use['time'].apply(lambda x: x.strftime("%H-%M"))
        df_use['time4'] = df_use['time4'].apply(lambda x: datetime.strptime(str(x), '%H-%M').hour)
        df_use['timeh'] = df_use['time'].apply(lambda x: x.strftime("%Y-%m-%d %H"))
        df_use['timeh'] = df_use['timeh'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H'))


        def sum_minutes(x):
            valores = []
            for e in horas4:
                valor = x - e
                if valor > 0:
                    valores.append(valor)
            if len(valores) > 0:
                return min(valores)
            else:
                return 0

        df_use['add'] = df_use.time4.apply(sum_minutes)

        df_use['4h'] = df_use.apply(lambda row: row['timeh'] - timedelta(hours=row['add']), axis=1 )

        return df_use


def combination_scores(indicadores_short, list_df_tickers, test=False, short=True):

    columnas5, columnas15, columnas1, columnas4 = [], [], [], []
    v_max = 0
    for indicator, values in indicadores_short.items():
        if (indicator == 'up_media5'):
            for v in values:
                columnas5.append(f'{indicator}_{v}')
                if v > v_max:
                    v_max = v
        elif (indicator == 'EMA5') or (indicator == 'volatility5') or (indicator == '5d+-') or (indicator == 'diff%') or (indicator == 'BB5'):
            for v in values:
                columnas5.append(f'{indicator}_{v}')
        elif (indicator == 'low5') or (indicator == 'high5') :
            columnas5.append(f'{indicator}')
        elif (indicator == 'rsi5') or (indicator == 'adx5'):
            for v in values:
                columnas5.append(f'{indicator}_{v}')
        elif (indicator == 'rsi15') or (indicator == 'adx15'):
            columnas15.append(f'{indicator}')
        elif (indicator == 'rsi1') or (indicator == 'adx1') or (indicator == '1d+-'):
            columnas1.append(f'{indicator}')
        elif (indicator == 'EMA15') or (indicator == 'BB15') or (indicator == '15d+-'):
            for v in values:
                columnas15.append(f'{indicator}_{v}')
        elif (indicator == 'EMA1') or (indicator == 'BB1') or (indicator == 'volatility1'):
            for v in values:
                columnas1.append(f'{indicator}_{v}')
        elif indicator == 'up_media1':
            for v in values:
                columnas1.append(f'{indicator}_{v}')
        elif indicator == 'EMA4':
            for v in values:
                columnas4.append(f'{indicator}_{v}')

    columnas = columnas5 + columnas15 + columnas1 + columnas4

    
    uniq = []
    for x in itertools.product([-1, 1],repeat= len(columnas) ) :
        uniq.append(x)
    uniques = np.array(uniq)
    columnas15.append('timehour')
    columnas1.append('timehour')
    columnas4.append('timehour')

    minute15_dfs = []
    for df_tick in list_df_tickers:
        df_use = df_tick.copy()
        df_use = df_use[df_use.timeframe == '15minute'].sort_values('time')
        df_use.rename(columns={'time':'timehour'}, inplace=True)
        for indicator, values in indicadores_short.items():
            if indicator == 'EMA15':
                for v in values:
                    df_use[f'{indicator}_{v}'] = np.where(df_use.close.ewm(span=v[0], adjust=False).mean() > df_use.close.ewm(span=v[1], adjust=False).mean(), 1, -1)
            elif indicator == 'rsi15':
                for v in values:
                    rsi = ta.momentum.rsi(close=df_use.close, window=v, fillna=True)
                    df_use['rsi'] = rsi
                    df_use[f'{indicator}'] = np.where(((df_use.rsi) < 30 & (df_use.rsi > 70)), 1, -1)
            elif indicator == 'adx15':
                for v in values:
                    adx = ta.trend.ADXIndicator(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use['adx'] = adx.adx()
                    df_use[f'{indicator}'] = np.where((df_use.adx > 30), 1, -1)
            elif indicator == '15d+-':
                for v in values:
                    adx = ta.trend.ADXIndicator(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use['d+'] = adx.adx_pos()
                    df_use['d-'] = adx.adx_neg()
                    df_use[f'{indicator}_{v}'] = np.where((df_use['d+'] > df_use['d-']), 1, -1)
            elif indicator == 'BB15':
                for v in values:
                    indicator_bb = ta.volatility.BollingerBands(close=df_use["close"], window=v, window_dev=2)
                    df_use['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
                    df_use['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
                    df_use[f'{indicator}_{v}'] = np.where((df_use.bb_bbhi > 0) | (df_use.bb_bbli > 0), 1, 0)
        minute15_dfs.append(df_use)

    hour1_dfs = []
    for df_tick in list_df_tickers:
        df_use = df_tick.copy()
        df_use = df_use[df_use.timeframe == '1hour'].sort_values('time')
        df_use.rename(columns={'time':'timehour'}, inplace=True)
        for indicator, values in indicadores_short.items():
            if indicator == 'EMA1':
                for v in values:
                    df_use[f'{indicator}_{v}'] = np.where(df_use.close.ewm(span=v[0], adjust=False).mean() > df_use.close.ewm(span=v[1], adjust=False).mean(), 1, -1)
            elif indicator == 'up_media1':
                for v in values:
                    df_use[f'{indicator}_{v}'] = np.where(df_use.close - df_use.close.ewm(span=v, adjust=False).mean() > 0, 1, -1)
            elif indicator == 'rsi1':
                for v in values:
                    rsi = ta.momentum.rsi(close=df_use.close, window=v, fillna=True)
                    df_use['rsi'] = rsi
                    df_use[f'{indicator}'] = np.where(((df_use.rsi) < 30 & (df_use.rsi > 70)), 1, -1)
            elif indicator == 'adx1':
                for v in values:
                    adx = ta.trend.ADXIndicator(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use['adx'] = adx.adx()
                    df_use[f'{indicator}'] = np.where((df_use.adx > 30), 1, -1)
            elif indicator == '1d+-':
                for v in values:
                    adx = ta.trend.ADXIndicator(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use['d+'] = adx.adx_pos()
                    df_use['d-'] = adx.adx_neg()
                    df_use[f'{indicator}'] = np.where((df_use['d+'] > df_use['d-']), 1, -1)
            elif indicator == 'BB1':
                for v in values:
                    indicator_bb = ta.volatility.BollingerBands(close=df_use["close"], window=v, window_dev=2)
                    df_use['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
                    df_use['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
                    df_use[f'{indicator}_{v}'] = np.where((df_use.bb_bbhi > 0) | (df_use.bb_bbli > 0), 1, 0)
            elif indicator == 'volatility1':
                for v in values:
                    volatility = ta.volatility.AverageTrueRange(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use['volatility'] = volatility.average_true_range() / df_use.close * 100
                    df_use[f'{indicator}_{v}'] = np.where((df_use['volatility'] > 0.8), 1, -1)
        hour1_dfs.append(df_use)

    hour4_dfs = []
    for df_tick in list_df_tickers:
        df_use = df_tick.copy()
        df_use = df_use[df_use.timeframe == '4hour'].sort_values('time')
        df_use.rename(columns={'time':'timehour'}, inplace=True)
        for indicator, values in indicadores_short.items():
            if indicator == 'EMA4':
                for v in values:
                    df_use[f'{indicator}_{v}'] = np.where(df_use.close.ewm(span=v[0], adjust=False).mean() > df_use.close.ewm(span=v[1], adjust=False).mean(), 1, -1)
        hour4_dfs.append(df_use)
    
        four = df_use.copy()
        four['time'] = pd.to_datetime(four['timehour'])
        four['time4'] = four['time'].apply(lambda x: x.strftime("%H-%M"))
        four['time4'] = four['time4'].apply(lambda x: datetime.strptime(str(x), '%H-%M').hour)
        hour4 = list(four.time4.unique())  
        
    target_dfs = [] 
    for i, df_tick in enumerate(list_df_tickers):

        df_use = df_tick.copy()
        df_use = df_use[df_use.timeframe == '5minute']
        df_use.sort_values('time', inplace=True)
        for indicator, values in indicadores_short.items():

            if indicator == 'up_media5':
                for v in values:
                    df_use[f'{indicator}_{v}'] = np.where(df_use.close - df_use.close.ewm(span=v, adjust=False).mean() > 0, 1, -1)
            elif indicator == 'EMA5':
                for v in values:
                    df_use[f'{indicator}_{v}'] = np.where(df_use.close.ewm(span=v[0], adjust=False).mean() > df_use.close.ewm(span=v[1], adjust=False).mean(), 1, -1)
                    if v[1] > v_max:
                        v_max = v[1]
            elif indicator == 'low5':
                df_use[f'{indicator}'] = np.where(df_use.close.rolling(values[0]).min() > df_use.close.rolling(values[1]).min(), 1, -1)
            elif indicator == 'high5':
                df_use[f'{indicator}'] = np.where(df_use.close.rolling(values[0]).max() < df_use.close.rolling(values[1]).max(), 1, -1)
            elif indicator == 'rsi5':
                for v in values:
                    rsi = ta.momentum.rsi(close=df_use.close, window=v, fillna=True)
                    df_use['rsi'] = rsi
                    df_use[f'{indicator}_{v}'] = np.where(((df_use.rsi) < 30 & (df_use.rsi > 70)), 1, -1)
            elif indicator == 'adx5':
                for v in values:
                    adx = ta.trend.ADXIndicator(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use['adx'] = adx.adx()
                    df_use[f'{indicator}_{v}'] = np.where((df_use.adx > 30), 1, -1)
            elif indicator == '5d+-':
                for v in values:
                    adx = ta.trend.ADXIndicator(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use['d+'] = adx.adx_pos()
                    df_use['d-'] = adx.adx_neg()
                    df_use[f'{indicator}_{v}'] = np.where((df_use['d+'] > df_use['d-']), 1, -1)
            elif indicator == 'volatility5':
                for v in values:
                    volatility = ta.volatility.AverageTrueRange(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    df_use['volatility'] = volatility.average_true_range() / df_use.close * 100
                    df_use[f'{indicator}_{v}'] = np.where((df_use['volatility'] > 0.8), 1, -1)
            elif indicator == 'diff%':
                for v in values:
                    df_use[f'{indicator}_{v}'] = np.where(((df_use.high.rolling(v).max() - df_use.close) / df_use.high.rolling(v).max() * 100) > 1, 1, -1)
            elif indicator == 'BB5':
                for v in values:
                    indicator_bb = ta.volatility.BollingerBands(close=df_use["close"], window=v, window_dev=2)
                    df_use['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
                    df_use['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
                    df_use[f'{indicator}_{v}'] = np.where((df_use.bb_bbhi > 0) | (df_use.bb_bbli > 0), 1, 0)


        df_use = df_use.iloc[v_max:, :].reset_index(drop=True)
        df_use['time'] = pd.to_datetime(df_use['time'])
        df_use['time1'] = df_use['time'].apply(lambda x: x.strftime("%Y-%m-%d %H"))
        df_use['time1'] = df_use['time1'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H'))

        df_use = convert_15(df_use)
        df_use = convert_4(df_use,hour4)

        df_use = df_use.merge(minute15_dfs[i][columnas15], left_on='15m', right_on='timehour', how='left')
        df_use = df_use.merge(hour1_dfs[i][columnas1], left_on='time1', right_on='timehour', how='left')
        df_use = df_use.merge(hour4_dfs[i][columnas4], left_on='4h', right_on='timehour', how='left')
        df_use.dropna(inplace=True)
        df_use.sort_values('time', inplace=True)
        df_use.reset_index(inplace=True)
        target_dfs.append(df_use)


    scores = np.zeros(uniques.shape[0])
    times = np.zeros(uniques.shape[0])

    dataframes_scores = []
    scores2 = []
    if not test:
        range_t = range(uniques.shape[0])
    else:
        range_t = test
    for e in range_t:
        wins_lose = np.array([0,0])
        wins_lose2 = []
        calcu_dfs = []
        
        for df_use in target_dfs:
            
            matri = df_use[columnas].values
            
            doty = np.dot(matri ,uniques[e].reshape(len(columnas), -1))
            df_use['target'] = np.where(doty == len(columnas), 1, 0)
            df_use['target2'] = df_use.target.shift(1).rolling(5).sum()
            df_use['target'] = np.where((df_use.target.shift(1).rolling(5).sum() > 0), 0, df_use['target'])
            df_use.dropna(inplace=True)
            df_use.reset_index(drop=True, inplace=True)

            if short:
                calculated, calculated_df = calculate_score_short(df_use)
            else:
                calculated, calculated_df = calculate_score_long(df_use)
            calculated_df['combi'] = e
            if test:
                calcu_dfs.append(calculated_df)
            wins_lose += calculated
            w, l = calculated[0], calculated[1]
            t = int(w+l)
            if t != 0:
                p = round(w/t*100,2)
            else:
                p = 0
            wins_lose2.append([p,t])

        if np.sum(wins_lose) == 0:
            scores[e] = 0
            times[e] = 0
            scores2.append(wins_lose2)
            continue
        
        wins = wins_lose[0]
        lose = wins_lose[1]
        score = round(wins / (wins + lose)*100,2)
        times_game = int(wins + lose)
        scores[e] = score
        times[e] = times_game
        scores2.append(wins_lose2)
        if test:
            dataframes_scores.append(pd.concat(calcu_dfs))

    scores_df = pd.DataFrame(uniques, columns=columnas)
    scores_df['score'] = scores
    scores_df['times'] = times
    if test:
        scores_df['ticks'] = ''
        scores_df.loc[test, 'ticks'] = scores2
    else:
        scores_df['ticks'] = scores2
    if test:
        try:
            dfs = pd.concat(dataframes_scores)
            dfs['timed'] = pd.to_datetime(dfs['time_sold']) - pd.to_datetime(dfs['time_buy'])
            dfs['timed'] = pd.to_timedelta(dfs['timed']).dt.total_seconds()/60
            dfs['hour_buy'] = pd.to_datetime(dfs['time_buy']).dt.hour
        except:
            time_buy = datetime.now()-timedelta(minutes=47)
            dfs = pd.DataFrame({'time_buy':[time_buy], 'price_buy':[0], 'time_sold':[time_buy], 'price_sold':[0], '%':[0]})
    else:
        dfs = pd.DataFrame()
    
    
    return scores_df, dfs


def calculate_score_short(df):  
            
    indice = np.argwhere(df.target.values == 1).reshape(1,-1).tolist()[0]
    buy = df.loc[indice, 'close'].to_list()
    time_buy = df.loc[indice, 'time'].to_list()

    
    lastbuy= False
    if indice and (indice[-1] == (df.shape[0] - 1)):
        lastidx = indice.pop(-1)
        lastbuy = buy.pop(-1)
        lasttime = time_buy.pop(-1)

    sold = []
    diff = []
    time_s = []
    max_high = []
    min_low = []
    wins, lose = 0,0
    for idx, precio in zip(indice, buy):
        done = False
        i = idx
        high = 0
        low = 0
        while not done:
            i += 1
            venta = df.loc[i, 'close']
            time_sold = df.loc[i, 'time']
            d = (venta - precio) / precio * 100
            if d > high:
                high = d
            if d < low:
                low = d
            if (d > 3):
                lose += 1
                diff.append(d)
                sold.append(venta)
                time_s.append(time_sold)
                max_high.append(high)
                min_low.append(low)
                break
            elif (d < -3):
                wins += 1
                time_s.append(time_sold)
                diff.append(d)
                sold.append(venta)
                max_high.append(high)
                min_low.append(low)
                break
            elif (i == (df.shape[0] - 1)):
                diff.append(d)
                sold.append(venta)
                time_s.append(time_sold)
                max_high.append(high)
                min_low.append(low)
                break

    if lastbuy:
        time_buy.append(lasttime)
        buy.append(lastbuy)
        time_s.append(lasttime)
        sold.append(lastbuy)
        max_high.append(0)
        min_low.append(0)
        diff.append(0)


    
    test = pd.DataFrame({'time_buy':time_buy, 'price_buy':buy, 'time_sold':time_s, 'price_sold':sold, '%':diff,'low':min_low, 'high':max_high})
    if df.shape[0] > 0:
        crypto = df.crypto.values[0]
    else:
        crypto = 'not'
    test['crypto'] = crypto
    

    return [wins,  lose], test

def calculate_score_long(df):  
            
    indice = np.argwhere(df.target.values == 1).reshape(1,-1).tolist()[0]
    buy = df.loc[indice, 'close'].to_list()
    time_buy = df.loc[indice, 'time'].to_list()

    
    lastbuy= False
    if indice and (indice[-1] == (df.shape[0] - 1)):
        lastidx = indice.pop(-1)
        lastbuy = buy.pop(-1)
        lasttime = time_buy.pop(-1)

    sold = []
    diff = []
    time_s = []
    min_low = []
    max_high = []
    wins, lose = 0,0
    for idx, precio in zip(indice, buy):
        done = False
        i = idx
        low = 0
        high = 0
        while not done:
            i += 1
            venta = df.loc[i, 'close']
            time_sold = df.loc[i, 'time']
            d = (venta - precio) / precio * 100
            if d < low:
                low = d
            if d > high:
                high = d
            if (d > 1):
                wins += 1
                diff.append(d)
                sold.append(venta)
                time_s.append(time_sold)
                min_low.append(low)
                max_high.append(high)
                break
            elif (d < -1):
                lose += 1
                time_s.append(time_sold)
                diff.append(d)
                sold.append(venta)
                min_low.append(low)
                max_high.append(high)
                break
            elif (i == (df.shape[0] - 1)):
                diff.append(d)
                sold.append(venta)
                time_s.append(time_sold)
                min_low.append(low)
                max_high.append(high)
                break

    if lastbuy:
        time_buy.append(lasttime)
        buy.append(lastbuy)
        time_s.append(lasttime)
        sold.append(lastbuy)
        min_low.append(0)
        max_high.append(0)
        diff.append(0)


    
    test = pd.DataFrame({'time_buy':time_buy, 'price_buy':buy, 'time_sold':time_s, 'price_sold':sold, '%':diff, 'low':min_low, 'high':max_high})
    if df.shape[0] > 0:
        crypto = df.crypto.values[0]
    else:
        crypto = 'not'
    test['crypto'] = crypto
    

    return [wins,  lose], test