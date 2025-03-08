import pandas as pd
import polars as pl
import numpy as np


# class Preprocessing():


#     def __init__(self,companies_df:pd.DataFrame,prices_df:pd.DataFrame):
#         self.companies_df=companies_df
#         self.prices_df=prices_df
    

def bulk_preprocessing(companies_df:pl.DataFrame, prices_df:pl.DataFrame, chosen:list):
    '''
    Preprocesses the data from simfin a and returns the necessary dataframe to train the model

    Parameters
    ----------

    companies_df : DataFrame with the companies information

    prices_df : DataFrame with the price information

    chosen : list containing the tickers from the desired companies

    Returns
    --------
    Modified and preprocessed DataFrame
    '''
    companies_df=companies_df.drop_nulls(subset=['Ticker','Company Name'])

    companies_df=companies_df.filter(pl.col('Ticker').is_in(chosen))
    prices_df=prices_df.filter(pl.col('Ticker').is_in(chosen))

    companies_df=companies_df.to_pandas()
    prices_df=prices_df.to_pandas()

    prc=prices_df.drop(['SimFinId','Dividend'],axis=1)

    comp=companies_df.drop(['SimFinId','Company Name', 'ISIN', 'End of financial year (month)', 
                    'Business Summary','Market','CIK','Main Currency'], axis=1)
        

    data=prc.merge(comp, how='left', on='Ticker')

    data['Date']=pd.to_datetime(data['Date'])
    data['Ticker']=data['Ticker'].astype('category')
    data.set_index('Date', inplace=True)
    data.columns=data.columns.str.replace('.','').str.replace(' ','_').str.replace(r'(?<=[a-zA-Z])([A-Z])', r'_\1', regex=True).str.lower()

    data['returns']=data['adj_close'].diff()
    data=data.dropna(subset=['returns'])

    return data



def streamed_preprocessing(companies_df:pl.DataFrame, prices_df:pd.DataFrame, chosen:list):
    '''
    Preprocesses the data from simfin a and returns the necessary dataframe to get the predictions

    Parameters
    ----------

    companies_df : DataFrame with the companies information

    prices_df : DataFrame with the price information

    chosen : list containing the tickers from the desired companies

    Returns
    --------
    Modified and preprocessed DataFrame
    '''
    companies_df=companies_df.to_pandas()
    companies_df=companies_df[companies_df['Ticker'].isin(chosen)]
    

    prc=prices_df.reset_index()
    prc=prc[prc['Ticker'].isin(chosen)]
    prc.drop(['SimFinId','Dividend'],axis=1, inplace=True)

    
    comp=companies_df.drop(['SimFinId','Company Name', 'ISIN', 'End of financial year (month)', 
                    'Business Summary','Market','CIK','Main Currency'], axis=1)
        

    data=prc.merge(comp, how='left', on='Ticker')

    data['Date']=pd.to_datetime(data['Date'])
    data['Ticker']=data['Ticker'].astype('category')
    data.set_index('Date', inplace=True)
    data.columns=data.columns.str.replace('.','').str.replace(' ','_').str.replace(r'(?<=[a-zA-Z])([A-Z])', r'_\1', regex=True).str.lower()

    return data

if __name__=='main':
    pass