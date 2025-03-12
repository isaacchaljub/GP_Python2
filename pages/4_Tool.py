import polars as pl
import streamlit as st
import plotly.express as px


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_data import COM, FinancialData


###########################################################

st.set_page_config(layout='wide', page_title="Trading Tool", page_icon="🔮")

st.markdown("# Try our trading tool for the stock you want")
st.write('''Currently we only support one-day forecasting given the constraints on Gradient Boosting Regression.
We advise you to use this forecast with caution, predicting stock returns is highly complicated and depends on
qualitative information not considered by the model''')
st.sidebar.header("Algorithmic trading tool")

def operate_trading_tool():
    comps = st.sidebar.selectbox("Select Company", COM['Company Name'].to_list())
    tk=COM.filter(pl.col('Company Name')==comps)['Ticker'].to_list()

    fp=FinancialData(tk)

    investing=fp.investing_strategy()

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
    # col1, col2, col3 = st.columns([1, 2, 1])  # Middle column (col2) will contain the text and DataFrame

    with st.container():
        #st.markdown('<p class="centered-text">📊 Your Data Overview</p>', unsafe_allow_html=True)
        st.write(investing)#, use_container_width=True)  # Displays DataFrame in a centered column
        #st.table(new_data)


if __name__ == "__main__":
    operate_trading_tool()