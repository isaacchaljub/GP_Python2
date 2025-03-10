import pandas as pd
import polars as pl
import numpy as np


from preprocessing import bulk_preprocessing, streamed_preprocessing


from xgboost import XGBRegressor
# from statsmodels.tsa.statespace.sarimax import SARIMAX

import simfin as sf
import os
from dotenv import load_dotenv
load_dotenv()

from time import sleep
import re

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


###########################################################

current_dir = os.path.dirname(os.path.abspath(__file__))

csv_path_companies = os.path.join(current_dir, 'us-companies.csv')
csv_path_prices = os.path.join(current_dir, 'us-shareprices-daily.csv')

COM=pl.read_csv(csv_path_companies, separator=';')
PRI=pl.read_csv(csv_path_prices, separator=';')
PRI=PRI.with_columns(pl.col('Date').str.to_datetime('%Y-%m-%d').cast(pl.Date))

COM=COM.drop_nulls(subset=['Ticker'])




class FinancialData():
    def __init__(self, chosen_companies:list=None, start_date:str=str(PRI['Date'].min()), end_date:str=str(PRI['Date'].max())):
        '''
        chosen_companies : List of the companies the analysis will be performed on.

        start_date : initial date of the historical data. If None, retrieves from the beginning of the available information. Data starts on '2019-04-08'

        end_date : final date of the historical data. If None, retrieves until the end of the available information. Data ends on '2024-03-11'
        
        '''
        
        date_format=re.compile(r'^\d{4}-\d{2}-\d{2}$')
        

        if not date_format.match(start_date):
            print("The start_date parameter must be a string passed in the format '%Y-%m-%d'")
        
        if not date_format.match(end_date):
            print("The end_date parameter must be a string passed in the format '%Y-%m-%d'")

        if len(chosen_companies)==0:
            print("The chosen companies' list cannot be empty")
        
        
        self.start_date=pd.to_datetime(start_date, format='%Y-%m-%d')       
        self.end_date=pd.to_datetime(end_date,format='%Y-%m-%d')
        self.chosen_companies=chosen_companies
        self.__api_key = os.getenv('api_key')
        self.companies, self.prices,  = self.__load_datasets__()
        self.new_data=None
        self.data=self.get_historical_data()
        self.updateable_data=self.get_historical_data()
        self.__model=self.__predictive_model__()
    
    

    def __load_datasets__(self):
        companies=COM
        prices=PRI

        #prices=prices.with_columns(pl.col('Date').str.to_datetime('%Y-%m-%d').cast(pl.Date))

        if self.start_date==PRI['Date'].min() and self.end_date==PRI['Date'].max(): #If no start or end date are specified
            prices=prices
        elif self.start_date==PRI['Date'].min() and self.end_date!=PRI['Date'].max(): #If no start date is specified
            prices=prices.filter(pl.col('Date')<=self.end_date)
        elif self.start_date!=PRI['Date'].min() and self.end_date==PRI['Date'].max(): #If no end date is specified
            prices=prices.filter(pl.col('Date')>=self.start_date)
        else: #If both dates are specified
            prices=prices.filter((pl.col('Date')>=self.start_date)&(pl.col('Date')<=self.end_date))

        return companies, prices
    
    
    def get_historical_data(self):
        '''
        Returns
        -------
        Dataframe with consolidated and preprocessed historical information
        '''
        return bulk_preprocessing(self.companies, self.prices, self.chosen_companies)
    
    def get_pl_sim(self):

        for stock in self.chosen_companies:
            data=self.data[self.data['ticker']==stock]
            start_date = data.index[0] #.strftime('%Y-%m-%d')
            end_date = data.index[-1] #.strftime('%Y-%m-%d')

            start_price=data['close'].loc[start_date]
            end_price=data['close'].loc[end_date]

            pl=np.round((end_price-start_price)/start_price,2)*100
            ret=np.round(end_price-start_price,2)

            if pl>0:
                return f"If you had bought one stock from {stock} at {self.start_date.date()} and sold at {self.end_date.date()}, you would have made ${ret}, a profit of {pl}%"
            elif pl<0:
                return f"If you had bought one stock from {stock} at {self.start_date.date()} and sold at {self.end_date.date()}, you would have lost ${ret}, a loss of {pl}%"
            else:
                return f"If you had bought one stock from {stock} at {self.start_date.date()} and sold at {self.end_date.date()}, you wouldn't have any profit or loss"
                



    
    def get_new_prices(self):
        '''
        Fetches new prices from the Simfin platform using the API

        Returns
        -------
        Dataframe with latest information (1 day) for every stock in the USA market
        '''

        sleep(0.5)
        sf.set_api_key(self.__api_key)
        sf.set_data_dir(os.getcwd()+'/streamed')

        stream=sf.load_shareprices(market='us',variant='latest');   

        new=streamed_preprocessing(self.companies, stream, self.chosen_companies)

        self.new_data=new
        return new

    def __predictive_model__(self):
        #Define target and exogenous variables
        data=self.updateable_data
        #df=data[data['ticker']==stock]
        x=data.drop('returns', axis=1)
        y=data['returns']


        model = XGBRegressor(objective='reg:squarederror', learning_rate=0.15, n_estimators=200, subsample=0.4, enable_categorical=True)
        model.fit(x, y)

        return model;
    

    def predict_new_return(self, stock_data):
        preds=self.__model.predict(stock_data)

        #self.__continuous_training__()

        return preds
    
    def investing_strategy(self):

        stocks=self.chosen_companies
        self.get_new_prices()
    
        for stock in stocks:
        
            pred=self.predict_new_return(self.new_data[self.new_data['ticker']==stock])
            historical_data=self.get_historical_data()
            rel=historical_data[historical_data['ticker']==stock]['returns']
            rang=rel.max()-rel.min()

            if pred>0.2*rang:
                #print(f'According to our model, the return tomorrow for {stock} will be greatly positive, you should buy')
                return f'According to our model, the return tomorrow for {stock} will be greatly positive, you should buy'
            elif pred <-0.2*rang:
                #print(f"According to our model, the return tomorrow for {stock} will be highly negative, you should sell")
                return f"According to our model, the return tomorrow for {stock} will be highly negative, you should sell"
            else:
                #print(f"According to our model, the return tomorrow for {stock} won't surpass 20% change in any direction, you should hold")
                return f"According to our model, the return tomorrow for {stock} won't surpass 20% change in any direction, you should hold"
        
        self.__continuous_training__()

    def __continuous_training__(self):
        '''
        Function to keep training the model with new predictions
        '''
        #Create new rows and add the returns to the dataframe
        aux=self.new_data.copy()
        aux['returns']=self.predict_new_return(aux)

        #Check if rows are not already in the data dataframe by their Date (index)
        if not aux.index.isin(self.updateable_data.index):
            self.updateable_data=pd.concat([self.updateable_data,aux])

        self.__model=self.__predictive_model__()
