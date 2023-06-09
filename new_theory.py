import pandas as pd
import numpy as np
import itertools
import ta
import time
import os
import json
import boto3
from datetime import datetime
from io import StringIO
from multiprocessing import Pool
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

## job name so we can identify partial results
JOB_NAME = os.environ.get('JOB_NAME')
JOB_NAME = 'tester3'
## total jobs mean how many jobs we want to submit, for us this mean in how many subspaces we want to divide our total space (total combinations)
number_of_jobs = os.environ.get('TOTAL_JOBS')
print("Number of jobs: ",number_of_jobs)
print('**********************************************')
# For testing locally we set a default value
if not number_of_jobs:
    number_of_jobs = 1
else:
    number_of_jobs = int(number_of_jobs)
# This index indicates which job is running the script. Each job will optimize a different space
array_job_idx = os.getenv("AWS_BATCH_JOB_ARRAY_INDEX")
print('im array number: ', array_job_idx)
# set a default value for testing
if not array_job_idx:
    array_job_idx = 0
else:
    array_job_idx = int(array_job_idx)
print('**********************************************')

# the dictionary containing the name of the indicator with its range of values to try
dictionary_string = os.getenv("DICTIONARY_STRING")
print('Dictionary string received! ' , dictionary_string)

print('**********************************************')
if not dictionary_string:
    dictionary_string = "{'EMA5':[15,30,45],'low5':[5,30], 'high5':[5,30], 'rsi5':[14], 'BB5':[10,20],'BB1':[20], 'rsi1':[14]}"

# df = pd.read_csv('btc.csv', index_col=0)


s3 = boto3.client('s3',
       aws_access_key_id='AKIA5EFPITXQ4GVIY7HN',
        aws_secret_access_key='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'
    )

BUCKET_NAME = 'awsbatchbacktester'
DATA_FILE_NAME = 'data/btc_usd.csv'
obj = s3.get_object(Bucket=BUCKET_NAME, Key=DATA_FILE_NAME)
df = pd.read_csv(obj['Body'], index_col=0)


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

    df_use = df_5minute.merge(df_hour[columnas_1], on='time1', how='left')

    return df_use


def calculate_score(df_use):  
    
    df = df_use.copy()
    df.reset_index(drop=True, inplace=True)

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
    volatility = 0
    for idx, precio in zip(indice, buy):
        done = False
        i = idx
        high = 0
        low = 0
        i_time_pass = 0
        while not done:
            i += 1
            i_time_pass += 1
            venta = df.loc[i, 'close']
            time_sold = df.loc[i, 'time']
            d = (venta - precio) / precio * 100
            if d > high:
                high = d
            if d < low:
                low = d
            if ((d > 1)|(d < -1)):
                volatility += 1
                diff.append(d)
                sold.append(venta)
                time_s.append(time_sold)
                max_high.append(high)
                min_low.append(low)
                break

            elif ((i_time_pass > 10)|(i == (df.shape[0] - 1))):
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
        crypto = 'btc'
    else:
        crypto = 'not'
    test['crypto'] = crypto
    

    return volatility, test


def score_multiprocessing(uniques, df_use):

    original_columns = ['time', 'open', 'high','low','close','volume','time1', 'adj_close', 'timeframe']
    columnas = len([col for col in df_use.columns if col not in original_columns])
    columnas_name = [col for col in df_use.columns if col not in original_columns]

    threshold = 100
    volatility, df_infos = [], []
    for unique_combi in tqdm(uniques):
        df_array = df_use.loc[:, columnas_name].values
        df_use_calculate = df_use.copy()
        zero_indices = [i for i in range(len(unique_combi)) if unique_combi[i] == 0]

        doty = np.dot(df_array ,unique_combi.reshape(columnas, -1))
        equal_to = columnas - len(zero_indices)
        df_use_calculate['target'] = np.where(doty == equal_to, 1, 0)
        df_use_calculate['target'] = np.where((df_use_calculate.target.shift(1).rolling(5).sum() > 0), 0, df_use_calculate['target'])
        if sum(df_use_calculate.target) > threshold:
            vol, df_info = calculate_score(df_use_calculate)
            df_info['combination'] = str(unique_combi)
            volatility.append([vol,sum(df_use_calculate.target)])
            df_infos.append(df_info)
        else:
         
            volatility.append([0,0])
            df_infos.append(pd.DataFrame())

    all_df = pd.DataFrame(volatility, columns=['vol','total'])

    unique_df = pd.DataFrame(uniques)
    unique_df['total'] = all_df['total']
    unique_df['vol'] = all_df['vol']
    unique_df['differ'] = round(unique_df.vol/unique_df.total * 100,2)

    unique_df = unique_df[unique_df.vol > 5].reset_index(drop=True)

    df_infos = [df_infos[e] for e in unique_df.sort_values('differ',ascending=False).head(15).index.to_list()]
    df_infos = pd.concat(df_infos)
    return [unique_df, df_infos]


def run_ghandi(df_use, uniques):

    # Initiate the multiprocessor    
    N_multiprocessors = 4 # In how many different processors we are going to run this index job. It will be changed to run different periods
    divide_number = int(uniques.shape[0] / N_multiprocessors)
    unique_list = [(uniques[divide_number*e:divide_number*(e+1)],df_use) for e in range(N_multiprocessors)]
    print('number of sub_spaces combinations: ', unique_list[0][0].shape[0])
    start_time = time.perf_counter()
    with Pool() as pool:
        result = pool.starmap(score_multiprocessing, unique_list)

    volatility, information = [],[]
    for i, r in enumerate(result):
        volatility.append(r[0])
        information.append(r[1])
    unique_df = pd.concat(volatility)
    information_df = pd.concat(information)
    return [unique_df, information_df]

if __name__ == "__main__":

    start_time = time.perf_counter()

    # In the following step we convert the dictionary string we loaded into a real dictionary
    # We create a list with each range of values for each indicator
    # Calculate total number of combinations, then divide the space into similar N size groups (N being total jobs)
    dictionary_string = dictionary_string.replace("'", "\"")
    # We must convert our parameters dictionary
    dictionary_string = json.loads(dictionary_string)

    df_use = create_combinations(dictionary_string, df)
    print(df_use.head(3))
    print(df_use.tail(3))
    original_columns = ['time', 'open', 'high','low','close','volume','time1', 'timeframe', 'adj_close']
    columnas = len([col for col in df_use.columns if col not in original_columns])

    uniq = []
    for x in itertools.product([-1, 1, 0],repeat= columnas ) :
        uniq.append(x)
    uniques = np.array(uniq)

    # Divide data into similar periods
    divide = int(uniques.shape[0] / number_of_jobs)
    # We will have a list with all the periods dataframes
    unique_divided = []
    for e in range(number_of_jobs):
        period_unique = uniques[divide*e:divide*(e+1),:]
        print(f'{e} len of combinations ->', period_unique.shape[0])
        unique_divided.append(period_unique)
        
    results = run_ghandi(df_use,unique_divided[array_job_idx])

    volatility = results[0]

    information = results[1]

    csv_buffer = StringIO()
    volatility.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    # Upload the CSV file to S3, this will be a partial result. We are optimizing one subspace.
    s3 = boto3.client(
        's3',
       aws_access_key_id='AKIA5EFPITXQ4GVIY7HN',
        aws_secret_access_key='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'
    )
    s3.put_object(Body=csv_content, Bucket='awsbatchbacktester', Key=f'partial_results/scores_{JOB_NAME}_volatility_{array_job_idx}.csv')
    
    csv_buffer = StringIO()
    information.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    s3.put_object(Body=csv_content, Bucket='awsbatchbacktester', Key=f'partial_results/scores_{JOB_NAME}_information_{array_job_idx}.csv')
    
    finish_time = time.perf_counter()
    print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
    print("---")
