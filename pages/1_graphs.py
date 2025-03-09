import polars as pl
import streamlit as st
import plotly.express as px

# from GP_PYTHON2 import 

#from GP_Python2.financial_data import COM, PRI, FinancialData

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_data import COM, PRI, FinancialData


###########################################################

st.set_page_config(layout='wide', page_title="Historical financial data", page_icon="📈")

#st.markdown("# Plotting Page")
st.sidebar.header("Plotting Page")


# @st.cache_data
def plot_stock_data():
    comps = st.sidebar.selectbox("Select Company", COM['Company Name'].to_list())
    #Pri=PRI.with_columns(pl.col('Date').str.to_datetime('%Y-%m-%d').cast(pl.Date))
    start= st.sidebar.date_input(label='start date',value=PRI['Date'].min())
    #start= st.sidebar.time_input(label='start date',value=None)

    end= st.sidebar.date_input(label='start date',value=PRI['Date'].max())
    #end= st.sidebar.time_input(label='end date',value=None)

    start_str = start.strftime('%Y-%m-%d')
    end_str   = end.strftime('%Y-%m-%d')
    #DESTINATION = st.sidebar.selectbox("Select Destination", destinations)
    tk=COM.filter(pl.col('Company Name')==comps)['Ticker'].to_list()

    #print(tk)
    fp=FinancialData(tk, start_date=start_str, end_date=end_str)
    data=fp.get_historical_data()

    ## PLOT FIGURE 1 ##
    fig1 = px.line(
        data,
        x=data.index,
        y="close",
        title=f"{tk[0]}",
        template="none",
    )

    fig1.update_xaxes(title="Date")
    fig1.update_yaxes(title="Closing Price")

    st.plotly_chart(fig1, use_container_width=True)



if __name__ == "__main__":
    # This is to configure some aspects of the app
    # st.set_page_config(
    #     layout="wide", page_title="Financial Market Analysis and Prediction Tool", page_icon="📈"
    # )

    # Write titles in the main frame and the side bar
    st.title("Historic Stock Prices")
    #st.sidebar.title("Select a Company")

    plot_stock_data()