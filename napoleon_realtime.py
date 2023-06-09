from time import sleep
import pandas as pd
from datetime import datetime, timedelta
import config as config
import itertools
import random
import numpy as np
from binance.client import Client
from binance.enums import *
# import simpleaudio as sa
from datetime import datetime 
import ta
import decimal
import tqdm
from helpers import *
from helper_napoleon import *
import warnings
warnings.filterwarnings('ignore')
import tweepy
 
# API keyws that yous saved earlier
api_key = "97r4ZfzENsl5CmPHD26nffhfX"
api_secrets = "B2JYXlsm1H4VW2pzK8klarhgOxqdH2pz6RDSacnARQ7sKfgJ0z"
access_token = "1549009688297181184-KtMzDikMjJjmpdddn6zbbLU76EiSLm"
access_secret = "AatmsmI42WlaD2FCUs5pVNL5CGCQ1HazLS8M8hihIitgz"
 
# Authenticate to Twitter
auth = tweepy.OAuthHandler(api_key,api_secrets)
auth.set_access_token(access_token,access_secret)
 
api = tweepy.API(auth)
# Create function to make the tweet notification
def tweet(tipo, result, result_price, symbol):
    random_tw = random.randint(1, 10000)
    status = f"{random_tw} | {tipo}  |  {symbol}  |  {result}  |  {result_price}%"
    api.update_status(status=status)


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

    def __init__(self, indicadores,cryptos, target_vector,trade_size, API_KEY, SECRET_KEY):
        
        # list of crypto symbols we want to trade (remember to use +USDT)
        self.cryptos = cryptos
        # Target vector is the combination of each indicator (optimized with aws batch jobs)
        self.target_vector = target_vector
        # This is the indicators dictionary used in the batch optimization
        self.list_indicators = indicadores
        # How much we want to spend in each trade
        self.trade_size = trade_size
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        self.client = Client(self.API_KEY, self.SECRET_KEY, tld='com')

        self.scores_dfs = {}        
        self.TRACK_DICTIONARY = {}
        # Dynamic dic will be used to move the stop loss once the profit is reached
        self.dynamic_dic = {}
        self.twitter_count = 0
        
        self.LAST_SELL = datetime.now() - timedelta(minutes=20)

    def run(self):

        self.client = Client(self.API_KEY, self.SECRET_KEY, tld='com')

        while True:
            
            # Keep local track, we can know when was stopped
            file1 = open("data/trackeo.txt","w")
            file1.write(f"{datetime.now()}")
            file1.close()
            
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
                tweet('ERROR', F'{datetime.now()}', '0',f'ERROR' )
                file1.write(f"{e},{datetime.now()}")
                file1.close()
                sleep(4)

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
                if crypto in self.TRACK_DICTIONARY.keys(): continue
                
                df_crypto = df_use_crypto[df_use_crypto.crypto == CRYPTO]
                strategy = check_strategy(self.target_vector, df_crypto)

                # When is the last price we have
                last_price_time = strategy.iloc[-1, 0]
                dif_time = (datetime.now() - last_price_time).total_seconds() / 60
                # Is the first signal?
                target = strategy.iloc[-1,-1]
                if target == 1:
                    random_tw = random.randint(1, 1000000)
                    tweet(f'{CRYPTO}', F'{datetime.now()}', '0',f'1st signal' )
                    file1 = open("data/trigger.txt","a")
                    file1.write(f"{CRYPTO},{datetime.now()}, {dif_time}\n")
                    file1.close()
                    sleep(1)

                if ((dif_time > 4) & (dif_time < 9)) & (target == 1):
        
                    time_elapsed = (datetime.now() - self.LAST_SELL).total_seconds() / 60
                    # If there are more than one trade signal at the same time
                    if (len(self.TRACK_DICTIONARY) < 4):
                        # Just in case..
                        self.client = Client(self.API_KEY, self.SECRET_KEY, tld='com')
                        # need to calculate the quantity given the amount of money we want for each trade
                        QUANTITY = self.trade_size / strategy.iloc[-1, 1]
                        # Let's grab the most recent price
                        historical = self.client.get_historical_klines(CRYPTO, '5m' , "2 days ago UTC")
                        minute5 = convert_df(historical)
                        LAST_PRICE = minute5.iloc[-1, 4]
                        # The following function will track the current price to see if we go short, long, or no trade.
                        LONG_OR_SHORT = self.wait_for_execution(crypto, LAST_PRICE, datetime.now())
                            
                        if LONG_OR_SHORT == 'SHORT':
                            order = self.sell_symbol_short(f'{crypto}',QUANTITY)
                            if order == False:
                                continue
                            else:
                                PRICE_SELL = float(order['fills'][0]['price'])
                                LOSE_PRICE = PRICE_SELL * 1.006
                                self.dynamic_dic[crypto] = float(PRICE_SELL) * 0.993
                                # Set a stop loss
                                stop_loss = self.create_stop_loss_short(f'{CRYPTO}', QUANTITY, LOSE_PRICE)
                                # Keep track of the trade
                                self.TRACK_DICTIONARY[CRYPTO] = {'crypto':CRYPTO, 'price_sell':PRICE_SELL,'quantity':QUANTITY, 'time_sell':datetime.now(), 'stop_loss_id':stop_loss, 'type':'SHORT'}
                                sleep(2)
        
                        if LONG_OR_SHORT == 'LONG':
                            order = self.buy_symbol(f'{crypto}',QUANTITY)
                            if order == False:
                                continue
                            else:
                                PRICE_BUY = float(order['fills'][0]['price'])
                                LOSE_PRICE = PRICE_BUY * 0.994
                                self.dynamic_dic[crypto] = float(PRICE_BUY) * 1.007
                                # Set a stop loss
                                stop_loss = self.create_stop_loss_long(f'{CRYPTO}', QUANTITY, LOSE_PRICE)
                                # Keep track of the trade
                                self.TRACK_DICTIONARY[CRYPTO] = {'crypto':CRYPTO, 'price_buy':PRICE_BUY,'quantity':QUANTITY, 'time_BUY':datetime.now(), 'stop_loss_id':stop_loss, 'type':'LONG'}
                                sleep(2)


                

                if len(self.TRACK_DICTIONARY) > 0:
                    # Track open trades
                    self.follow_orders_short()
                    self.follow_orders_long()

                        
            time_elapsed_total = (datetime.now() - self.LAST_SELL).total_seconds() / 60

            if (int(time_elapsed_total) % 120) == 0: 
                if int(time_elapsed_total) != 0:
                    random_tw = random.randint(1, 1000000)
                    tweet('STILL ALIVE', 'be patient bro', '0',f'small steps' )
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
        

        PRICE_long = PRICE * 1.003
        PRICE_short = PRICE * 0.997
    
        DONE = False
        while not DONE:
            sleep(2)
            historical = self.client.get_historical_klines(crypto, '1m' , "5 minutes ago UTC")
            last_price = convert_df(historical)
            last_price = last_price.iloc[-1, -2]
            time_elapsed = (datetime.now() - TIME).total_seconds() / 60
            if time_elapsed > 20:
                DONE = True
                return False
            elif last_price > PRICE_long:
                DONE = True
                return 'LONG'
            elif last_price < PRICE_short:
                DONE = True
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
    
    def finish_order_short(self,SYMBOL):

        QUANTITY = self.TRACK_DICTIONARY[SYMBOL]['quantity'] * 1.003

        QUANTITY_BUY = self.create_quantity(SYMBOL, QUANTITY)

        o = self.buy_symbol(SYMBOL, QUANTITY_BUY)

        FINAL_PRICE = o['fills'][0]['price']

        stop_loss_id = self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id']
        self.client.cancel_margin_order(
                symbol=SYMBOL,
                orderId=stop_loss_id)

        self.pay_loan(SYMBOL, QUANTITY)

        return FINAL_PRICE
    
    def finish_order_long(self,SYMBOL, cancelled=False):

        QUANTITY = self.TRACK_DICTIONARY[SYMBOL]['quantity'] * 0.998

        QUANTITY_SELL = self.create_quantity(SYMBOL, QUANTITY)
        print('from_system', QUANTITY_SELL)
        
        info = self.client.get_margin_account()
        for assets in info['userAssets']:
            if assets['asset'] == SYMBOL[:-4]:
                QUANTITY_SELL2 = assets['netAsset']
                print('from_api', QUANTITY_SELL2)
        
        stop_loss_id = self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id']
        print('ANTES DE CANCELAR LA ORDEN')
        
        if not cancelled:
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
            file1 = open("trackeo_long.txt","w")
            file1.write(f"antes de crear el stop, {SYMBOL}, {datetime.now()}")
            file1.close()
            stop_loss = self.create_stop_loss_long(f'{SYMBOL}', QUANTITY, PRICE)
            file1 = open("trackeo_long.txt","w")
            file1.write(f"despues de crear el stop, {SYMBOL}, {datetime.now()}")
            file1.close()
        
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
                historical = self.client.get_historical_klines(SYMBOL, '1m' , "1 hour ago UTC")
                last_price = convert_df(historical)
                last_price = last_price.iloc[-1, -2]
            except:
                break

            status = self.client.get_margin_order(
            symbol=SYMBOL,
            orderId= self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id'] )['status']

            LOSE_PRICE = float(self.client.get_margin_order(
            symbol=SYMBOL,
            orderId= self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id'] )['price'])

            tipo = self.TRACK_DICTIONARY[SYMBOL]['type']

            if time_elapsed > 50:
                print(f'SYMBOL {SYMBOL} NO MORE TIME :( short')
                FINAL_PRICE = float(self.finish_order_short(SYMBOL))
                result_price = round((PRICE_SELL - FINAL_PRICE) / FINAL_PRICE * 100, 2)
                tweet(tipo, 'Time Elapsed :|', result_price, SYMBOL)
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
                    print(f'{SYMBOL} LOST :(')
                tweet(tipo, resu, result_price, SYMBOL)
            elif last_price > LOSE_PRICE * 1.002:
                print(f'SYMBOL {SYMBOL} STOP DIDNT WORK :( short')
                FINAL_PRICE = float(self.finish_order_short(SYMBOL))
                result_price = round((PRICE_SELL - FINAL_PRICE) / FINAL_PRICE * 100, 2)
                tweet(tipo, 'LOST :(', result_price, SYMBOL)
                DONE = True
            elif last_price < self.dynamic_dic[SYMBOL]:
                print('dynamic short')
                print(f'SYMBOL {SYMBOL} WIN! :)')
                sleep(25)
                try:
                    self.dynamic_stop(SYMBOL, self.dynamic_dic[SYMBOL], SHORT=True)
                    self.dynamic_dic[SYMBOL] = self.dynamic_dic[SYMBOL] * 0.997
                except:
                    FINAL_PRICE = float(self.finish_order_short(SYMBOL))
                    result_price = round((PRICE_SELL - FINAL_PRICE) / FINAL_PRICE * 100, 2)
                    tweet(tipo, 'WIN :)', result_price, SYMBOL)
                    DONE = True


            if DONE:
                track = pd.read_csv('data/track_df.csv', index_col=0)
                tipo = 'SHORT'
                combi = self.TRACK_DICTIONARY[SYMBOL]['combi']
                TRACK_DF = pd.DataFrame({'CRYPTO':SYMBOL, 'SELL_TIME':TIME_SELL,  'PRICE_SELL':PRICE_SELL,'BUY_TIME':datetime.now(), 'PRICE_BUY':FINAL_PRICE, 'type':tipo}, index=[0])
                TRACK_DF = pd.concat([track, TRACK_DF])
                TRACK_DF.to_csv('data/track_df.csv')
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
                    historical = self.client.get_historical_klines(SYMBOL, '1m' , "15 minute ago UTC")
                    last_price = convert_df(historical)
                    last_price = last_price.iloc[-1, -2]
                except:
                    break

                tipo = self.TRACK_DICTIONARY[SYMBOL]['type']

                status = self.client.get_margin_order(
                symbol=SYMBOL,
                orderId= self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id'] )['status']

                LOSE_PRICE = float(self.client.get_margin_order(
                symbol=SYMBOL,
                orderId= self.TRACK_DICTIONARY[SYMBOL]['stop_loss_id'] )['price'])

                if time_elapsed > 50:
                    FINAL_PRICE = float(self.finish_order_long(SYMBOL))
                    result_price = round((FINAL_PRICE - BUY_PRICE) / BUY_PRICE * 100, 2)
                    tweet(tipo, 'Time Elapsed :|', result_price, SYMBOL)
                    print(f'SYMBOL {SYMBOL} NO MORE TIME :(, FINISH ORDER')
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
                    tweet(tipo, resu, result_price, SYMBOL)
                elif last_price < LOSE_PRICE * 0.998:
                    print(f'SYMBOL {SYMBOL} STOP DIDNT WORK :(')
                    FINAL_PRICE = float(self.finish_order_long(SYMBOL))
                    result_price = round((FINAL_PRICE - BUY_PRICE) / BUY_PRICE * 100, 2)
                    tweet(tipo, 'LOST :(', result_price, SYMBOL)
                    DONE = True
                elif last_price > self.dynamic_dic[SYMBOL]:
                    print('paso el win')
                    DYNAMIC = True
                    sleep(25)
                    try:
                        self.dynamic_stop(SYMBOL, self.dynamic_dic[SYMBOL])
                        self.dynamic_dic[SYMBOL] = self.dynamic_dic[SYMBOL] * 1.0035
                    except:
                        FINAL_PRICE = float(self.finish_order_long(SYMBOL,cancelled=True))
                        result_price = round((FINAL_PRICE - BUY_PRICE) / BUY_PRICE * 100, 2)
                        tweet(tipo, 'WIN :)', result_price, SYMBOL)
                        DONE = True
            if DONE:
                track = pd.read_csv('data/track_df.csv', index_col=0)
                tipo = 'LONG'
                TRACK_DF = pd.DataFrame({'CRYPTO':SYMBOL, 'SELL_TIME':datetime.now(),  'PRICE_SELL':FINAL_PRICE,'BUY_TIME':TIME_BUY, 'PRICE_BUY':BUY_PRICE, 'type':tipo}, index=[0])
                TRACK_DF = pd.concat([track, TRACK_DF])
                TRACK_DF.to_csv('data/track_df.csv')
                del self.TRACK_DICTIONARY[SYMBOL]
                break
                print('*****************************************************')
                print('*****************************************************')


 
    def pay_loan(self, SYMBOL, QUANTITY):
        
        QUANTITY = QUANTITY * 0.998
        QUANTITY_INT = self.create_quantity(SYMBOL, QUANTITY)
#         info = self.client.get_margin_account()
#         for assets in info['userAssets']:
#             if assets['asset'] == SYMBOL[:-4]:
#                 QUANTITY_INT = assets['netAsset']
#                 print(QUANTITY_INT)

        transaction = self.client.repay_margin_loan(asset=SYMBOL[:-4], amount=QUANTITY_INT)
        transaction_id = transaction['tranId']
        
        sleep(5)

        details = self.client.get_margin_repay_details(asset=SYMBOL[:-4], txId=transaction_id)
        if details['rows'][0]['status'] != 'CONFIRMED':
            return print('ERROR PAYING THE LOAN')
        
        
    
                

    def last_results(self):
        return self.scores_dfs



            




