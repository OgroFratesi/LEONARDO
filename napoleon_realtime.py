from time import sleep
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
from binance.client import Client
from requests_oauthlib import OAuth1Session
from binance.enums import *
from datetime import datetime 
import decimal
from helper_napoleon import *
import warnings
import boto3
warnings.filterwarnings('ignore')
 
# API keyws that yous saved earlier
consumer_key = "97r4ZfzENsl5CmPHD26nffhfX"
consumer_secret = "B2JYXlsm1H4VW2pzK8klarhgOxqdH2pz6RDSacnARQ7sKfgJ0z"
access_token = "1549009688297181184-KtMzDikMjJjmpdddn6zbbLU76EiSLm"
access_token_secret = "AatmsmI42WlaD2FCUs5pVNL5CGCQ1HazLS8M8hihIitgz"

# Make the request
oauth = OAuth1Session(
    consumer_key,
    client_secret=consumer_secret,
    resource_owner_key=access_token,
    resource_owner_secret=access_token_secret,
)

# Create function to make the tweet notification
def tweet(tipo, result, result_price, symbol, time_elapsed):
    random_tw = random.randint(1, 10000)
    status = f"{symbol} | {tipo}  |  {time_elapsed}  |  {result}  |  {result_price}%"
    # Making the request
    payload = {'text':status}
    response = oauth.post(
        "https://api.twitter.com/2/tweets",
        json=payload,
    )



'''
The following class will request the lately historical prices for the given crypto symbols and add the given indicators. 
After that, it will iterate over each of them, checking if at least one indicator is giving the first signal. Then, that crypto 
will enter into the second step. It is going to look into the next prices and it will compare it with the signal price. If the 
current price goes above 0.3% or below -0.3%, it will buy (or sell) the crypto.
Now, after each iteration, it will check the price and the stop loss for the crypto. If the price goes above 0.7% for long or 
below -0.7% for shorts, the stop loss will move to this price and the next profit price will be 0.3% more than the one before.

To sum up:

1- Request historial prices
2- Add indicators
3- Check first signal
4- Look the current price against the signal price for Long or Short
5- In case the trade is made, keep tracking the price until the trade is over
'''
class LEONARDO:

    def __init__(self, indicadores,cryptos, target_vector,trade_size,API_KEY, SECRET_KEY):

        # or create a resource
        dynamodb_resource = boto3.resource('dynamodb',region_name='eu-west-2')
        dynamodb_table = dynamodb_resource.Table('leonardo_results')
        
        self.dynamodb_table = dynamodb_table
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        
        # list of crypto symbols we want to trade (remember to use +USDT)
        self.cryptos = cryptos
        # Target vector is the combination of each indicator (optimized with aws batch jobs)
        self.target_vector = target_vector
        # This is the indicators dictionary used in the batch optimization
        self.list_indicators = indicadores
        # How much we want to spend in each trade
        self.trade_size = trade_size
        self.client = Client(self.API_KEY, self.SECRET_KEY, tld='com')

        self.scores_dfs = {}        
        self.TRACK_DICTIONARY = {}
        self.wait_for_execution_dic = {}
        # Dynamic dic will be used to move the stop loss once the profit is reached
        self.dynamic_dic = {}
        self.twitter_count = 0
        
        self.LAST_SELL = datetime.now() - timedelta(minutes=20)
        


    def run(self):

        self.client = Client(self.API_KEY, self.SECRET_KEY, tld='com')

        while True:
            
            
            # We connect every time we can in case we lost connection
            self.client = Client(self.API_KEY, self.SECRET_KEY, tld='com')
            test_df = []
                
            try:

                for cr in self.cryptos:

                    historical = self.client.get_historical_klines(cr, '1h' ,"10 days ago UTC")
                    hour1 = convert_df(historical)
                    hour1['timeframe'] = '1hour'

                    historical = self.client.get_historical_klines(cr, '5m' , "3 days ago UTC")
                    minute5 = convert_df(historical)
                    minute5['timeframe'] = '5minutes'

                    this_crypto = pd.concat([hour1, minute5])
                    this_crypto['crypto'] = cr
                    test_df.append(this_crypto)

            except Exception as e: 
                
                file1 = open("data/error.txt","w")
                # Send tweet in case it was disconnected
                tweet('ERROR', f'{datetime.now()}', f'{e}',f'ERROR', 'ERROR')
                file1.write(f"{e},{datetime.now()}")
                file1.close()
                sleep(4)
                continue

            df_all = pd.concat(test_df)
            # Add indicators to each historical crypto price
            df_use_crypto = []
            for crypto in self.cryptos:
                df_crypto = df_all[df_all.crypto == crypto]
                df_use = create_combinations(self.list_indicators, df_crypto)
                df_use_crypto.append(df_use)
            df_use_crypto = pd.concat(df_use_crypto)
            

            for CRYPTO in self.cryptos:
                
                # If we already have an active trade with this crypto, continue with the second one
                if CRYPTO in self.TRACK_DICTIONARY.keys(): continue
                
                df_crypto = df_use_crypto[df_use_crypto.crypto == CRYPTO]
                strategy = check_strategy(self.target_vector, df_crypto)
                
                # When is the last price we have
                last_price_time = strategy.iloc[-2, 0]
                dif_time = (datetime.now() - last_price_time).total_seconds() / 60
                # Is the first signal?
                target = strategy.iloc[-2,-1]
                price = strategy.iloc[-2, 4]
                if ((dif_time > 4) & (dif_time < 9)) & (target == 1):
                    # If there are more than one trade signal at the same time
                    if (len(self.TRACK_DICTIONARY) < 2)&(CRYPTO not in self.wait_for_execution_dic.keys()):
                        sleep(1)
                        
                        # Just in case..
                        self.client = Client(self.API_KEY, self.SECRET_KEY, tld='com')
                        # need to calculate the quantity given the amount of money we want for each trade
                        QUANTITY = self.trade_size / strategy.iloc[-1, 1]
                        # Let's grab the most recent price
                        historical = self.client.get_historical_klines(CRYPTO, '5m' , "1 hour ago UTC")
                        minute5 = convert_df(historical)
                        LAST_PRICE = minute5.iloc[-2, 4]
                        # Insert symbol into the wait for execution dictionary
                        self.wait_for_execution_dic[CRYPTO] = {'SYMBOL':CRYPTO, 'QUANTITY':QUANTITY, 'LAST_PRICE':LAST_PRICE, 'TIME':datetime.now()}
                        self.insert_item_dynamo_signal(last_price_time, CRYPTO, 'signal', LAST_PRICE)

                if CRYPTO in self.wait_for_execution_dic.keys():

                    LAST_PRICE = self.wait_for_execution_dic[CRYPTO]['LAST_PRICE']
                    QUANTITY = self.wait_for_execution_dic[CRYPTO]['QUANTITY']
                    TIME = self.wait_for_execution_dic[CRYPTO]['TIME']
                    # The following function will track the current price to see if we go short, long, or no trade.
                    LONG_OR_SHORT = self.wait_for_execution(CRYPTO, LAST_PRICE, TIME)
                        
                    if LONG_OR_SHORT == 'SHORT':
                        sleep(4)
                        order = self.sell_symbol_short(f'{CRYPTO}',QUANTITY)
                        if order == False:
                            continue
                        else:
                            PRICE_SELL = float(order['fills'][0]['price'])
                            # tweet(f'{CRYPTO}', f'{datetime.now()}', f'{PRICE_SELL}',f'SHORT')
                            LOSE_PRICE = PRICE_SELL * 1.005
                            self.dynamic_dic[CRYPTO] = float(PRICE_SELL) * 0.991
                            # Set a stop loss. If there is an error, the price went quickly into the lose boundary, take the loss.
                            try:
                                stop_loss = self.create_stop_loss_short(f'{CRYPTO}', QUANTITY, LOSE_PRICE)
                            except:
                                self.TRACK_DICTIONARY[CRYPTO] = {'crypto':CRYPTO, 'price_sell':PRICE_SELL,'quantity':QUANTITY, 'time_sell':datetime.now(), 'stop_loss_id':'none', 'type':'SHORT'}
                                self.finish_order_short(CRYPTO,cancelled=True)
                                del self.TRACK_DICTIONARY[CRYPTO]
                                continue
                            # Keep track of the trade
                            self.TRACK_DICTIONARY[CRYPTO] = {'crypto':CRYPTO, 'price_sell':PRICE_SELL,'quantity':QUANTITY, 'time_sell':datetime.now(), 'stop_loss_id':stop_loss, 'type':'SHORT'}
                            sleep(2)

                            
    
                    if LONG_OR_SHORT == 'LONG':
                        order = self.buy_symbol(f'{CRYPTO}',QUANTITY)
                        sleep(1)
                        if order == False:
                            continue
                        else:
                            PRICE_BUY = float(order['fills'][0]['price'])
                            # tweet(f'{CRYPTO}', f'{datetime.now()}', f'{PRICE_BUY}',f'LONG')
                            LOSE_PRICE = PRICE_BUY * 0.995
                            self.dynamic_dic[CRYPTO] = float(PRICE_BUY) * 1.009
                            # Set a stop loss. If there is an error, the price went quickly into the lose boundary, take the loss.
                            try:
                                stop_loss = self.create_stop_loss_long(f'{CRYPTO}', QUANTITY, LOSE_PRICE)
                            except:
                                self.TRACK_DICTIONARY[CRYPTO] = {'crypto':CRYPTO, 'price_buy':PRICE_BUY,'quantity':QUANTITY, 'time_buy':datetime.now(), 'stop_loss_id':'none', 'type':'LONG'}
                                self.finish_order_long(self,CRYPTO, cancelled=True)
                                del self.TRACK_DICTIONARY[CRYPTO]
                                continue
                            # Keep track of the trade
                            self.TRACK_DICTIONARY[CRYPTO] = {'crypto':CRYPTO, 'price_buy':PRICE_BUY,'quantity':QUANTITY, 'time_buy':datetime.now(), 'stop_loss_id':stop_loss, 'type':'LONG'}
                            sleep(2)

            

                if len(self.TRACK_DICTIONARY) > 0:
                    # Track open trades
                    self.follow_orders_short()
                    self.follow_orders_long()

                        
            time_elapsed_total = (datetime.now() - self.LAST_SELL).total_seconds() / 60

            if (int(time_elapsed_total) % 180) == 0: 
                if int(time_elapsed_total) != 0:
                    random_number = random.randint(1, 10000)
                    tweet('STILL ALIVE', 'be patient bro', '0',f'small steps', random_number)
                    self.LAST_SELL = datetime.now()

            
            self.client.close_connection()

  
    def create_quantity(self, SYMBOL, QUANTITY):

        info = self.client.get_symbol_info(SYMBOL)
        sleep(3)
        for e in range(len(info['filters'])):
            if info['filters'][e]['filterType'] == 'LOT_SIZE':
                deci = info['filters'][e]['stepSize']
                deci = str(float(deci))

        d = decimal.Decimal(deci)
        d = d.as_tuple().exponent * -1  

        if d == 0:
            NEW_QUANTITY = int(QUANTITY)
        elif deci == '1.0':
            NEW_QUANTITY = int(QUANTITY)
        else:
            NEW_QUANTITY = round(QUANTITY, d)

        return NEW_QUANTITY

    def create_price(self, SYMBOL, PRICE):

        info = self.client.get_symbol_info(SYMBOL)
        sleep(3)
        for e in range(len(info['filters'])):
            if info['filters'][e]['filterType'] == 'PRICE_FILTER':
                deci = info['filters'][e]['tickSize']
                deci = str(float(deci))

        d = decimal.Decimal(deci)
        d = d.as_tuple().exponent * -1  

        if d == 0:
            NEW_PRICE = int(PRICE)
        else:    
            NEW_PRICE = round(PRICE, d)

        return NEW_PRICE
    
    def wait_for_execution(self, crypto, PRICE, TIME):
        
        PRICE_long = PRICE * 1.002
        PRICE_short = PRICE * 0.998

        print('Entered wait_for_execution..')
        sleep(2)
        historical = self.client.get_historical_klines(crypto, '1m' , "15 minutes ago UTC")
        last = convert_df(historical)
        last_price = last.iloc[-1, 4]
        time_elapsed = (datetime.now() - TIME).total_seconds() / 60

        if time_elapsed > 20:
            del self.wait_for_execution_dic[crypto]
            return False
        elif last_price > PRICE_long:
            del self.wait_for_execution_dic[crypto]
            return 'LONG'
        elif last_price < PRICE_short:
            del self.wait_for_execution_dic[crypto]
            return 'SHORT'
        
            

            
           
    def sell_symbol_short(self,SYMBOL, QUANTITY):

        QUANTITY = self.create_quantity(SYMBOL, QUANTITY)
        
        try:
            transaction = self.client.create_margin_loan(asset=SYMBOL[:-4], amount=QUANTITY)
        except:
            return False
        
        transaction_id = transaction['tranId']
        sleep(4)
        details = self.client.get_margin_loan_details(asset=SYMBOL[:-4], txId=transaction_id)

        if details['rows'][0]['status'] != 'CONFIRMED':
            return print('ERROR IN LOAN')
        else:
            print('LOAN SUCCESFUL')
            
        print('QUANTITY: ', QUANTITY)

        order = self.client.create_margin_order(
            symbol=SYMBOL,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=QUANTITY,
            )
        print(f'{SYMBOL} SOLD ')
        
        return order

    def buy_symbol(self,SYMBOL, QUANTITY):
        
        QUANTITY = self.create_quantity(SYMBOL, QUANTITY)

        order = self.client.create_margin_order(
            symbol=SYMBOL,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=QUANTITY,
            )
        
        return order

    def create_stop_loss_short(self, SYMBOL, QUANTITY, LOSE_PRICE):
        
        QUANTITY = self.create_quantity(SYMBOL, QUANTITY)
        LOSE_PRICE = self.create_price(SYMBOL, LOSE_PRICE)
        
        order = self.client.create_margin_order(
            symbol=SYMBOL,
            side=SIDE_BUY,
            type=ORDER_TYPE_STOP_LOSS_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=QUANTITY,
            price=LOSE_PRICE,
            stopPrice=LOSE_PRICE
            )

        print('STOP LOSS READY.', order['orderId'])

        return order['orderId']
    
    
    def create_stop_loss_long(self, SYMBOL, QUANTITY, LOSE_PRICE):
        QUANTITY = QUANTITY * 0.998
        QUANTITY = self.create_quantity(SYMBOL, QUANTITY)
        LOSE_PRICE = self.create_price(SYMBOL, LOSE_PRICE)
        
        order = self.client.create_margin_order(
            symbol=SYMBOL,
            side=SIDE_SELL,
            type=ORDER_TYPE_STOP_LOSS_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=QUANTITY,
            price=LOSE_PRICE,
            stopPrice=LOSE_PRICE
            )

        print('STOP LOSS READY.',order['orderId'] )

        return order['orderId']
    
    def finish_order_short(self,SYMBOL, cancelled=False):

        QUANTITY = self.TRACK_DICTIONARY[SYMBOL]['quantity'] * 1.003

        QUANTITY_BUY = self.create_quantity(SYMBOL, QUANTITY)

        o = self.buy_symbol(SYMBOL, QUANTITY_BUY)

        FINAL_PRICE = o['fills'][0]['price']
        
        if not cancelled:
            stop_loss_id = self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id']
            self.client.cancel_margin_order(
                    symbol=SYMBOL,
                    orderId=stop_loss_id)

        self.pay_loan(SYMBOL, QUANTITY)

        return FINAL_PRICE
    
    def finish_order_long(self,SYMBOL, cancelled=False):

        QUANTITY = self.TRACK_DICTIONARY[SYMBOL]['quantity'] * 0.999

        QUANTITY_SELL = self.create_quantity(SYMBOL, QUANTITY)
        print('from_system', QUANTITY_SELL)
        
        info = self.client.get_margin_account()
        for assets in info['userAssets']:
            if assets['asset'] == SYMBOL[:-4]:
                QUANTITY_SELL2 = assets['netAsset']
                print('from_api', QUANTITY_SELL2)
        
        
        if not cancelled:

            stop_loss_id = self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id']
            print('ANTES DE CANCELAR LA ORDEN')
            self.client.cancel_margin_order(
                    symbol=SYMBOL,
                    orderId=stop_loss_id)
        

        order = self.client.create_margin_order(
            symbol=SYMBOL,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=QUANTITY_SELL,
            )
        
        print(f'{SYMBOL} SOLD ')
        

        FINAL_PRICE = order['fills'][0]['price']

        return FINAL_PRICE

        
    def dynamic_stop(self, SYMBOL, PRICE, SHORT=False):
        
        sleep(1)
        
        QUANTITY = self.TRACK_DICTIONARY[SYMBOL]['quantity']
        
        stop_loss_id = self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id']
       
        
        if SHORT:
            stop_loss = self.create_stop_loss_short(f'{SYMBOL}', QUANTITY, PRICE)
            self.client.cancel_margin_order(
                symbol=SYMBOL,
                orderId=stop_loss_id)

        else:
            self.client.cancel_margin_order(
                symbol=SYMBOL,
                orderId=stop_loss_id)
            sleep(2)

            stop_loss = self.create_stop_loss_long(f'{SYMBOL}', QUANTITY, PRICE)

        
        self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id'] = stop_loss

    def follow_orders_short(self):

        for SYMBOL in self.TRACK_DICTIONARY.keys():

            if self.TRACK_DICTIONARY[SYMBOL]['type'] == 'LONG':
                continue
            DONE = False
                
            sleep(5)
            
            TIME_SELL = self.TRACK_DICTIONARY[SYMBOL]['time_sell']
            PRICE_SELL = self.TRACK_DICTIONARY[SYMBOL]['price_sell']
            LOSE_PRICE = self.TRACK_DICTIONARY[SYMBOL]['price_sell'] * 1.006
            QUANTITY = self.TRACK_DICTIONARY[SYMBOL]['quantity']

            time_elapsed = (datetime.now() - TIME_SELL).total_seconds() / 60
            
            try:
                historical = self.client.get_historical_klines(SYMBOL, '5m' , "30 minutes ago UTC")
                last = convert_df(historical)
                last_price = last.iloc[-1, -2]
                last_time = last.iloc[-1, 0]
                
#                 dif_time_last = (datetime.now() - last_time).total_seconds() / 60
#                 if dif_time_last < 4:
#                     continue
            except:
                break

            status = self.client.get_margin_order(
            symbol=SYMBOL,
            orderId= self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id'] )['status']

            LOSE_PRICE = float(self.client.get_margin_order(
            symbol=SYMBOL,
            orderId= self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id'] )['price'])

            tipo = self.TRACK_DICTIONARY[SYMBOL]['type']

            if time_elapsed > 30:
                FINAL_PRICE = float(self.finish_order_short(SYMBOL))
                result_price = round((PRICE_SELL - FINAL_PRICE) / FINAL_PRICE * 100, 2)
                tweet(tipo, 'Time Elapsed :|', result_price, SYMBOL, round(time_elapsed,2))
                DONE = True
            elif status in ['FILLED', 'CANCELED']:
                self.pay_loan(SYMBOL, QUANTITY)
                DONE = True
                FINAL_PRICE = LOSE_PRICE
                result_price = round((PRICE_SELL - FINAL_PRICE) / FINAL_PRICE * 100, 2)
                if result_price > 0:
                    resu = 'WIN :)'
                else:
                    resu = 'LOST :('
                tweet(tipo, resu, result_price, SYMBOL, round(time_elapsed,2))
            elif last_price > LOSE_PRICE * 1.002:
                FINAL_PRICE = float(self.finish_order_short(SYMBOL))
                result_price = round((PRICE_SELL - FINAL_PRICE) / FINAL_PRICE * 100, 2)
                tweet(tipo, 'LOST :(', result_price, SYMBOL, round(time_elapsed,2))
                DONE = True
            elif last_price < self.dynamic_dic[SYMBOL]:
                sleep(25)
                try:
                    self.dynamic_stop(SYMBOL, self.dynamic_dic[SYMBOL], SHORT=True)
                    self.dynamic_dic[SYMBOL] = self.dynamic_dic[SYMBOL] * 0.997
                except:
                    FINAL_PRICE = float(self.finish_order_short(SYMBOL))
                    result_price = round((PRICE_SELL - FINAL_PRICE) / FINAL_PRICE * 100, 2)
                    tweet(tipo, 'WIN :)', result_price, SYMBOL, round(time_elapsed,2))
                    DONE = True


            if DONE:
                tipo = 'SHORT'
                self.insert_item_dynamo_final(datetime.now(),SYMBOL,tipo,datetime.now(),FINAL_PRICE, TIME_SELL, PRICE_SELL)
                del self.TRACK_DICTIONARY[SYMBOL]
                break
                print('*****************************************************')
                print('*****************************************************')
                
    def follow_orders_long(self):

        for SYMBOL in self.TRACK_DICTIONARY.keys():
            if self.TRACK_DICTIONARY[SYMBOL]['type'] == 'SHORT':
                continue   

            DONE = False
            
            while not DONE:
                TIME_BUY = self.TRACK_DICTIONARY[SYMBOL]['time_buy']
                BUY_PRICE = self.TRACK_DICTIONARY[SYMBOL]['price_buy']
                QUANTITY = self.TRACK_DICTIONARY[SYMBOL]['quantity']

                time_elapsed = (datetime.now() - TIME_BUY).total_seconds() / 60
                
                try:
                    historical = self.client.get_historical_klines(SYMBOL, '5m' , "30 minutes ago UTC")
                    last = convert_df(historical)
                    last_price = last.iloc[-1, -2]
                    last_time = last.iloc[-1, 0]

#                     dif_time_last = (datetime.now() - last_time).total_seconds() / 60
#                     if dif_time_last < 4:
#                         continue
                except:
                    break

                tipo = self.TRACK_DICTIONARY[SYMBOL]['type']

                status = self.client.get_margin_order(
                symbol=SYMBOL,
                orderId= self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id'] )['status']

                LOSE_PRICE = float(self.client.get_margin_order(
                symbol=SYMBOL,
                orderId= self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id'] )['price'])

                if time_elapsed > 30:
                    FINAL_PRICE = float(self.finish_order_long(SYMBOL))
                    result_price = round((FINAL_PRICE - BUY_PRICE) / BUY_PRICE * 100, 2)
                    tweet(tipo, 'Time Elapsed :|', result_price, SYMBOL, round(time_elapsed,2))
                    DONE = True
                elif status in ['FILLED', 'CANCELED']:
                    DONE = True
                    FINAL_PRICE = LOSE_PRICE
                    result_price = round((FINAL_PRICE - BUY_PRICE) / BUY_PRICE * 100, 2)
                    if result_price > 0:
                        resu = 'WIN :)'
                    else:
                        print(f'{SYMBOL} LOST :(')
                        resu = 'LOST :('
                    tweet(tipo, resu, result_price, SYMBOL, round(time_elapsed,2))
                elif last_price < LOSE_PRICE * 0.998:
                    FINAL_PRICE = float(self.finish_order_long(SYMBOL))
                    result_price = round((FINAL_PRICE - BUY_PRICE) / BUY_PRICE * 100, 2)
                    tweet(tipo, 'LOST :(', result_price, SYMBOL, round(time_elapsed,2))
                    DONE = True
                elif last_price > self.dynamic_dic[SYMBOL]:
                    DYNAMIC = True
                    sleep(25)
                    try:
                        self.dynamic_stop(SYMBOL, self.dynamic_dic[SYMBOL])
                        self.dynamic_dic[SYMBOL] = self.dynamic_dic[SYMBOL] * 1.0035
                    except:
                        FINAL_PRICE = float(self.finish_order_long(SYMBOL,cancelled=True))
                        result_price = round((FINAL_PRICE - BUY_PRICE) / BUY_PRICE * 100, 2)
                        tweet(tipo, 'WIN :)', result_price, SYMBOL, round(time_elapsed,2))
                        DONE = True
            if DONE:
                tipo = 'LONG'
                self.insert_item_dynamo_final(datetime.now(),SYMBOL,tipo,TIME_BUY,BUY_PRICE, datetime.now(), FINAL_PRICE)
                del self.TRACK_DICTIONARY[SYMBOL]
                break
                print('*****************************************************')
                print('*****************************************************')


 
    def pay_loan(self, SYMBOL, QUANTITY):
        
        QUANTITY = QUANTITY * 0.998
        QUANTITY_INT = self.create_quantity(SYMBOL, QUANTITY)

        transaction = self.client.repay_margin_loan(asset=SYMBOL[:-4], amount=QUANTITY_INT)
        transaction_id = transaction['tranId']
        
        sleep(5)

        details = self.client.get_margin_repay_details(asset=SYMBOL[:-4], txId=transaction_id)
        if details['rows'][0]['status'] != 'CONFIRMED':
            return print('ERROR PAYING THE LOAN')
        
    def insert_item_dynamo_final(self, time, symbol, TYPE, buy_time, buy_price, sell_time, sell_price):

        item_data = {
            'time': f'{time}',
            'symbol': f'{symbol}',
            'type':f'{TYPE}',
            'buy_time': f'{buy_time}',
            'price_buy':f'{buy_price}',
            'sell_time':f'{sell_time}',
            'sell_price':f'{sell_price}'
        }

        response = self.dynamodb_table.put_item(Item=item_data)

    def insert_item_dynamo_signal(self, time, symbol, TYPE, price_signal):

        item_data = {
            'time': f'{time}',
            'symbol': f'{symbol}',
            'type':f'{TYPE}',
            'price_signal': f'{price_signal}'
        }

        response = self.dynamodb_table.put_item(Item=item_data)



            




