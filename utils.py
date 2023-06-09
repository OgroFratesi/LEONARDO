import pandas as pd
import time
import boto3
import itertools
import ta
from datetime import datetime
import warnings
from tqdm import tqdm
import numpy as np

''' In this script we have 5 functions, the first three are used to divide the total space into small subspaces.
The following function, collect_results, will gather all the partial results for each job and concatenate the results in one df.
The last function, submit_job, will submit all the jobs we want using the desire parameters. '''

def count_combinations(list_params):
    ''' This function just calculate all the possible combinations of parameters'''

    combinations = 1
    for p in list_params:
        combinations *= len(p)
    return combinations

def divide_space_first(list_params):


    # For each list inside our list, take the longest list index
    max_len_p = 0
    for i, p in enumerate(list_params):
        if len(p) > max_len_p:
            max_len_p = len(p)
            param_to_devide = i
    # halve the selected list
    list_to_divide_into = int(len(list_params[param_to_devide])/2)

    # Now there are two lists
    new_params = list_params[param_to_devide][:list_to_divide_into], list_params[param_to_devide][list_to_divide_into:]

    # This is tricky
    # original list -> [[indicator_1_range_values],[indicator_2_range_values],[indicator_3_range_values]]
    # if indicator 3 is the longest list, the results will be two list:
    # list a) [[indicator_1_range_values],[indicator_2_range_values],[ 1st HALF indicator_3_range_values]]
    # list b) [[indicator_1_range_values],[indicator_2_range_values],[ 2st HALF indicator_3_range_values]]

    list_list_params = []
    for i_new_version in range(2):
        new_list = []
        for i, p in enumerate(list_params):
            if i == param_to_devide:
                new_list.append(new_params[i_new_version])
            else:
                new_list.append(p)
        list_list_params.append(new_list)

    return list_list_params


def divide_space_second(list_params, times=2):

    # call the divide space as many times we want to divide our total space
    times_divided=1
    while times_divided < times:
        max_mult = 0
        if times_divided == 1:
            list_params = divide_space_first(list_params)
            times_divided += 1
        else:
            # after dividing the original list into 2 subspaces
            for i,p in enumerate(list_params):
                mult = 1
                for e in p:
                    mult *= len(e)
                # The following space we want to halve will be the one with more combinations
                if mult > max_mult:
                    max_mult = mult
                    list_to_divide = i
            # list_params_new is the result of dividing the space (already a subspace), again into other two sub_subspaces.
            list_params_new = divide_space_first(list_params[list_to_divide])
            # Dont need anymore the subspace, we have two sub_subspaces of it now.
            list_params.pop(list_to_divide)
            # Add the two sub_subspaces to the original list
            list_params += list_params_new
            
            times_divided += 1
            

    return list_params



## ******************************************************************************************************* ##

# We create the functions for submitting the batch job, wait for it, and then collect the results from s3

# Collect results function that we will use after the job is done

def collect_results(JOB_NAME, info=False):

    # Set up S3 client
    s3 = boto3.client('s3',
       aws_access_key_id='AKIA5EFPITXQ4GVIY7HN',
        aws_secret_access_key='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'
    )

    # Define S3 bucket and prefix where CSV files are stored
    bucket_name = 'awsbatchbacktester'
    prefix = 'partial_results/'

    # Get list of CSV files in the bucket with the specified prefix
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    csv_files = [obj['Key'] for obj in objects['Contents'] if (JOB_NAME in obj['Key'] and 'information' in obj['Key'])]
    if not info: csv_files = [obj['Key'] for obj in objects['Contents'] if (JOB_NAME in obj['Key'] and 'volatility' in obj['Key'])]
    
    # Load CSV files into a list of dataframes
    dfs = []
    for file in csv_files:
        obj = s3.get_object(Bucket=bucket_name, Key=file)
        dfs.append(pd.read_csv(obj['Body']))

    # Concatenate dataframes into a single dataframe
    df = pd.concat(dfs, ignore_index=True)

    # Display the resulting dataframe
    return df

def submit_job(DICTIONARY_STRING,
               TOTAL_JOBS = "4", 
               JOB_DEFINITION = 'arn:aws:batch:us-west-1:902310698465:job-definition/batch_tutorial_job_definition:6', # ARN for the job difinition (always the same)
               JOB_QUEUE = 'arn:aws:batch:us-west-1:902310698465:job-queue/batch_tutorial_queue', # ARN for the job queue (always the same)
               JOB_NAME = 'testing_new_day_8', # Could be anything, final and partial files will contain this name
               N_CPUS = 1,
               TIMEOUT=6000,
               ):

    batch = boto3.client('batch', region_name='us-west-1',
       aws_access_key_id='AKIA5EFPITXQ4GVIY7HN',
        aws_secret_access_key='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'
    )


    job_timeout = TIMEOUT  # we can estimate this number for each job

    response = batch.submit_job(
    jobName=JOB_NAME,
    jobQueue=JOB_QUEUE,
    jobDefinition=JOB_DEFINITION,
    arrayProperties= { "size": int(TOTAL_JOBS) },
    timeout={
        'attemptDurationSeconds': job_timeout
    },
    containerOverrides={ 'environment': [
            {
                'name': 'DICTIONARY_STRING',
                'value': DICTIONARY_STRING,
            },
            {
                'name': 'JOB_NAME',
                'value': JOB_NAME,
            },
            {
                'name': 'TOTAL_JOBS',
                'value': TOTAL_JOBS,
            }
        ],
         "resourceRequirements": [
                                {
                                "type": "VCPU",
                                "value": N_CPUS
                                }
                                            ], }
        )

    job_id = response['jobId']

    print(f'Job submitted with ID: {job_id}')


    # wait and check every N seconds if the job is ready
    DONE = False
    n_try = 0
    while not DONE:
        
        time.sleep(15)
        
        response = batch.describe_jobs(jobs=[job_id])

        job_status = response['jobs'][0]['status']

        if job_status == 'SUCCEEDED':
            print('Job succeeded')
            DONE = True
            # df = collect_results(job_id, JOB_NAME)
            return 'DONE'
        elif job_status == 'FAILED':
            print('Job failed')
            DONE = True
            return pd.DataFrame()
        elif job_status == 'RUNNING':
            if n_try % 10 == 0:
                print('Job is still running')  
        else:
            if n_try % 10 == 0:
                print(f'Job status is {job_status}')

        n_try += 1

        if n_try == 1500:
            DONE = True



# Create the dictionary containing each indicator name and the range or list of values for each, here is an example:
# Use is as default
fast_choice = [x for x in range(2, 20, 2)]
slow_choice = [x for x in range(25, 76, 3)]
rsi = [x for x in range(10,30, 2)]
boll_period = [x for x in range(14,30,4)]
boll_devfactor = [1,2,3]

DICTIONARY = {"fast_lenght":fast_choice, "slow_lenght":slow_choice, "rsi":rsi, "boll_period":boll_period, "boll_devfactor":boll_devfactor}
# We need it as string
dictionary_string_test = f"{DICTIONARY}"



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
                    # df_use['rsi'] = rsi
                    df_use[f'{indicator}_{v}'] = np.where(((rsi) < 30 & (rsi > 70)), 1, -1)
            elif (indicator == 'adx5')|(indicator == 'adx1'):
                for v in values:
                    adx = ta.trend.ADXIndicator(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    # df_use['adx'] = adx.adx()
                    df_use[f'{indicator}_{v}'] = np.where((adx.adx() > 30), 1, -1)
            elif indicator == '5d+-':
                for v in values:
                    adx = ta.trend.ADXIndicator(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    # df_use['d+'] = adx.adx_pos()
                    # df_use['d-'] = adx.adx_neg()
                    df_use[f'{indicator}_{v}'] = np.where((adx.adx_pos() > adx.adx_neg()), 1, -1)
            elif (indicator == 'volatility5')|((indicator == 'volatility1')):
                for v in values:
                    volatility = ta.volatility.AverageTrueRange(high=df_use.high, low=df_use.low, close=df_use.close, window=v, fillna=True)
                    # df_use['volatility'] = volatility.average_true_range() / df_use.close * 100
                    df_use[f'{indicator}_{v}'] = np.where(((volatility.average_true_range() / df_use.close * 100) > 0.8), 1, -1)
            elif indicator == 'diff%':
                for v in values:
                    df_use[f'{indicator}_{v}'] = np.where(((df_use.high.rolling(v).max() - df_use.close) / df_use.high.rolling(v).max() * 100) > 2, 1, -1)
            elif (indicator == 'BB5')|(indicator == 'BB1'):
                for v in values:
                    indicator_bb = ta.volatility.BollingerBands(close=df_use["close"], window=v, window_dev=2)
                    # df_use['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
                    # df_use['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
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


def score_multiprocessing(uniques, df_use, threshold=100):

    original_columns = ['crypto','time', 'open', 'high','low','close','volume','time1', 'adj_close', 'timeframe']
    columnas = len([col for col in df_use.columns if col not in original_columns])
    columnas_name = [col for col in df_use.columns if col not in original_columns]

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

    unique_df = unique_df[unique_df.vol > 5]

    df_infos = [df_infos[e] for e in unique_df.sort_values('differ',ascending=False).head(15).index.to_list()]
    if df_infos:
        df_infos = pd.concat(df_infos)
    return [unique_df, df_infos]


def calculate_score(df_use):  
    df = df_use.copy()
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
    high_first_list, low_first_list = [],[]
    volatility = 0
    for idx, precio in zip(indice, buy):
        done = False
        i = idx
        high = 0
        low = 0
        i_time_pass = 0
        FIRST = 1
        high_first, low_first = 0,0
        while not done:
            i += 1
            i_time_pass += 1
            venta = df.loc[i, 'close']
            time_sold = df.loc[i, 'time']
            d = (venta - precio) / precio * 100
            if d > high:
                high = d
                if (high > 0.3)&(FIRST):
                    high_first = FIRST
                    FIRST = False
            if d < low:
                low = d
                if (low < -0.3)&(FIRST):
                    low_first = FIRST
                    FIRST = False
            if ((d > 1)|(d < -1)):
                volatility += 1
                diff.append(d)
                sold.append(venta)
                time_s.append(time_sold)
                max_high.append(high)
                min_low.append(low)
                high_first_list.append(high_first)
                low_first_list.append(low_first)
                break

            elif ((i_time_pass > 10)|(i == (df.shape[0] - 1))):
                diff.append(d)
                sold.append(venta)
                time_s.append(time_sold)
                max_high.append(high)
                min_low.append(low)
                high_first_list.append(high_first)
                low_first_list.append(low_first)
                break

    if lastbuy:
        time_buy.append(lasttime)
        buy.append(lastbuy)
        time_s.append(lasttime)
        sold.append(lastbuy)
        max_high.append(0)
        min_low.append(0)
        diff.append(0)
        high_first_list.append(high_first)
        low_first_list.append(low_first)


    
    test = pd.DataFrame({'time_buy':time_buy, 'price_buy':buy, 'time_sold':time_s, 'price_sold':sold, '%':diff,'low':min_low, 'high':max_high,
                         'high_first':high_first_list,'low_first':low_first_list})
    if df.shape[0] > 0:
        crypto = 'btc'
    else:
        crypto = df.crypto.values[0]
    test['crypto'] = crypto
    

    return volatility, test

def run_ghandi(df_use, uniques, threshold=100):

    # Initiate the multiprocessor    
    N_multiprocessors = 4 # In how many different processors we are going to run this index job. It will be changed to run different periods
    divide_number = int(uniques.shape[0] / N_multiprocessors)
    unique_list = [(uniques[divide_number*e:divide_number*(e+1)],df_use) for e in range(N_multiprocessors)]
    print('number of sub_spaces combinations: ', unique_list[0][0].shape[0])

    unique_df = score_multiprocessing(uniques, df_use, threshold=threshold)

    return [unique_df[0],unique_df[1]]

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



