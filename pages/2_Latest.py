import polars as pl
import streamlit as st
import plotly.express as px

# from GP_PYTHON2 import 

#from GP_Python2.financial_data import COM, PRI, FinancialData

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_data import COM, FinancialData


###########################################################

st.set_page_config(layout='wide', page_title="Most recent financial data", page_icon="📈")

st.markdown("# Latest stock information")
st.sidebar.header("Latest stock information Page")

def retrieve_stock_data():
    comps = st.sidebar.selectbox("Select Company", COM['Company Name'].to_list())
    tk=COM.filter(pl.col('Company Name')==comps)['Ticker'].to_list()

    fp=FinancialData(tk,None,None)

    new_data=fp.get_new_prices()

    # Set page layout
    #st.set_page_config(page_title="Centered Text & DataFrame", layout="wide")

    # Custom CSS for centering
    st.markdown(
        """
        <style>
            .centered-text {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                color: #4CAF50;
            }
            .dataframe-container {
                display: flex;
                justify-content: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Centering using st.columns()
    #col1, col2, col3 = st.columns([1, 2, 1])  # Middle column (col2) will contain the text and DataFrame

    with st.container():
        #st.markdown('<p class="centered-text">📊 Your Data Overview</p>', unsafe_allow_html=True)
        st.dataframe(new_data, use_container_width=True)  # Displays DataFrame in a centered column
        #st.table(new_data)


if __name__ == "__main__":
    retrieve_stock_data()