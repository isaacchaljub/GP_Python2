import pandas as pd
import polars as pl
import numpy as np



from preprocessing import bulk_preprocessing, streamed_preprocessing


from xgboost import XGBRegressor
#from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA as am

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
    def __init__(self, chosen_companies:list, start_date:str=str(PRI['Date'].min()), end_date:str=str(PRI['Date'].max())):
        '''
        chosen_companies : List of the companies the analysis will be performed on.

        start_date : initial date of the historical data. If None, retrieves from the beginning of the available information. Data starts on '2019-04-08'

        end_date : final date of the historical data. If None, retrieves until the end of the available information. Data ends on '2024-03-11'
        
        '''
        
        date_format=re.compile(r'^\d{4}-\d{2}-\d{2}$')
        

        if not date_format.match(start_date):
            raise ValueError("The start_date parameter must be a string passed in the format '%Y-%m-%d'")
        
        if not date_format.match(end_date):
            raise ValueError("The end_date parameter must be a string passed in the format '%Y-%m-%d'")

        if len(chosen_companies)==0:
            raise ValueError("The chosen companies' list cannot be empty")
        
        try:
            self.start_date=pd.to_datetime(start_date, format='%Y-%m-%d')       
            self.end_date=pd.to_datetime(end_date,format='%Y-%m-%d')

            if self.end_date < self.start_date:
                raise ValueError('end_date can not be earlier than start date')

            self.chosen_companies=chosen_companies
            self.__api_key = os.getenv('api_key')
            self.companies, self.prices,  = self.__load_datasets__()
            self.new_data=None
            self.data=self.get_historical_data()
            self.updateable_data=self.get_historical_data()
            self.__model=self.__predictive_model__()

        except Exception as e:
            print(f'There was an error on the initiation: {e}')
    

    def __load_datasets__(self):
        try:
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
        
        except Exception as e:
            print(f'Error Loading datasets:{e}')
    
    
    def get_historical_data(self):
        '''
        Returns
        -------
        Dataframe with consolidated and preprocessed historical information
        '''
        return bulk_preprocessing(self.companies, self.prices, self.chosen_companies)
    
    def get_pl_sim(self):

        try:
            for stock in self.chosen_companies:
                data=self.data[self.data['ticker']==stock]
                start_date = data.index[0] #.strftime('%Y-%m-%d')
                end_date = data.index[-1] #.strftime('%Y-%m-%d')

                start_price=data['close'].loc[start_date]
                end_price=data['close'].loc[end_date]

                pl=np.round(100*(end_price-start_price)/start_price,2)
                ret=np.round(end_price-start_price,2)

                if pl>0:
                    return f"If you had bought one stock from {stock} at {self.start_date.date()} and sold at {self.end_date.date()}, you would have made ${ret:.2f}, a profit of {pl:.2f}%"
                elif pl<0:
                    return f"If you had bought one stock from {stock} at {self.start_date.date()} and sold at {self.end_date.date()}, you would have lost ${ret:.2f}, a loss of {pl:.2f}%"
                else:
                    return f"If you had bought one stock from {stock} at {self.start_date.date()} and sold at {self.end_date.date()}, you wouldn't have any profit or loss"
        except Exception as e:
            print(f'Error: {e}')



    
    def get_new_prices(self):
        '''
        Fetches new prices from the Simfin platform using the API

        Returns
        -------
        Dataframe with latest information (1 day) for every stock in the USA market
        '''
        try:
            sleep(0.5)
            sf.set_api_key(self.__api_key)
            sf.set_data_dir(os.getcwd()+'/streamed')

            stream=sf.load_shareprices(market='us',variant='latest');   

            new=streamed_preprocessing(self.companies, stream, self.chosen_companies)

            self.new_data=new
            return new
        except Exception as e:
            print(f'Error:{e}')

    def __predictive_model__(self,data=None, start_date=None, end_date=None):
        try:
            if start_date is None:
                start_date=self.start_date
            
            if end_date is None:
                end_date=self.end_date

            if data is None:
                data=self.data
            #Define target and exogenous variables
            data=data.loc[start_date:end_date]
            # data=data.as_freq
            #df=data[data['ticker']==stock]
            # x=data.drop('returns', axis=1)
            y=data['returns']


            # model = XGBRegressor(objective='reg:squarederror', learning_rate=0.15, n_estimators=200, subsample=0.4, enable_categorical=True)
            # model.fit(x, y)
            model=am(y,order=(1,0,1))
            result=model.fit()

            return result
    
        except Exception as e:
            print(f'Setting Predictive Model Error:{e}')
    

    # def predict_new_return(self, stock_data):
    #     preds=self.__model.predict(stock_data)

    #     #self.__continuous_training__()

    #     return preds

    def convert_from_log_to_return(self, starting_price, log_preds):
        preds=[]
        last=starting_price

        for pred in log_preds:
            ret=last*(np.exp(pred)-1)
            preds.append(ret)
            last+=ret

        return preds

    
    def predictions(self, start_date:str, end_date:str):
        
        try:        
            start=pd.to_datetime(start_date)
            end=pd.to_datetime(end_date)
            steps=(end-start).days

            forecast_data=self.data.loc[start- pd.Timedelta(days=1):end]

            closes=forecast_data['close'].diff()
            actual=closes.dropna()

            prices=self.data.loc[start - pd.Timedelta(days=1), 'close']
            res=self.__predictive_model__(None,self.start_date, start)

            # preds=pd.DataFrame(data=model.predict(passed_data),index=passed_data.index,columns=['returns'])
            #passed_returns=pd.DataFrame(data=passed_returns, index=passed_data.index, columns=['returns'])
            log_preds=res.forecast(steps=steps)
            
            preds_list=self.convert_from_log_to_return(prices.iloc[-1], log_preds)

            future_dates = pd.date_range(start=start, periods=steps, freq='B')
            preds = pd.Series(preds_list, index=future_dates)
            

            return actual, preds
        
        except Exception as e:
            print(f'Error when predicting various returns: {e}')


    def investing_strategy(self, steps):

        
        stocks=self.chosen_companies

        for stock in stocks:
            
            data=self.data[self.data['ticker']==stock]
            res=self.__predictive_model__(data,self.start_date, self.end_date)
            #self.__continuous_training__()
            log_preds=res.forecast(steps=steps)
            preds_list=self.convert_from_log_to_return(data['close'].iloc[-1], log_preds)

            future_dates = pd.date_range(start=self.end_date, periods=steps, freq='B')
            preds = pd.Series(preds_list, index=future_dates)

            min_pred=preds.min()
            date_min_pred=preds.idxmin()
            max_pred=preds.max()
            date_max_pred=preds.idxmax()

            print(preds)

            if date_min_pred>date_max_pred:
                return preds, f"It seems like the stock will be bearish for some time, you shouldn't buy now"
            else:
                return preds, f"To obtain the maximum profit in the timeframe, you should buy on {date_min_pred} and sell on {date_max_pred}, for a forecasted return of ${max_pred-min_pred}"

                # if pred>0.2*rang:
                #     #print(f'According to our model, the return tomorrow for {stock} will be greatly positive, you should buy')
                #     return f'According to our model, the return tomorrow for {stock} will be greatly positive, you should buy'
                # elif pred <-0.2*rang:
                #     #print(f"According to our model, the return tomorrow for {stock} will be highly negative, you should sell")
                #     return f"According to our model, the return tomorrow for {stock} will be highly negative, you should sell"
                # else:
                #     #print(f"According to our model, the return tomorrow for {stock} won't surpass 20% change in any direction, you should hold")
                #     return f"According to our model, the return tomorrow for {stock} won't surpass 20% change in any direction, you should hold"
        # except Exception as e:
        #     print(f'Ivesting Strategy Error:{e}')
        

    # def __continuous_training__(self):
    #     '''
    #     Function to keep training the model with new predictions
    #     '''
    #     #Create new rows and add the returns to the dataframe
    #     aux=self.new_data.copy()
    #     aux['returns']=self.predict_new_return(aux)

    #     #Check if rows are not already in the data dataframe by their Date (index)
    #     if not aux.index.isin(self.updateable_data.index):
    #         self.updateable_data=pd.concat([self.updateable_data,aux])

    #     self.__model=self.__predictive_model__()


# if __name__=='main':
#     fp=FinancialData(COM['Company Name'])
#     fp.get_new_prices()