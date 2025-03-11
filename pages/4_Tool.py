import polars as pl
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_data import COM, FinancialData


###########################################################

st.set_page_config(layout='wide', page_title="Trading Tool", page_icon="🔮")

st.markdown("# Try our trading tool for the stock you want")
st.sidebar.header("Algorithmic trading tool")

def operate_trading_tool():
    try:
        comps = st.sidebar.selectbox("Select Company", COM['Company Name'].to_list())
        tk=COM.filter(pl.col('Company Name')==comps)['Ticker'].to_list()

        steps=int(st.text_input(label='Please specify a number of days to forecast the returns. The default is 5', value=5))

        

        fp=FinancialData(tk)
        preds, investing=fp.investing_strategy(steps)
        data=fp.data

        fig = go.Figure()

        fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["returns"],
            mode='lines',
            name='Actual',
            line=dict(color='blue')  # optionally specify color
            )
        )

        # Add second line (e.g., 'preds')
        fig.add_trace(
            go.Scatter(
                x=preds.index,
                y=preds.values,
                mode='lines',
                name='Prediction',
                line=dict(color='red')   # optionally specify color
            )
        )

        # Configure layout
        fig.update_layout(
            title=f"{tk[0]}",
            template="none",
            xaxis_title="Date",
            yaxis_title="Returns",
        )

        st.plotly_chart(fig, use_container_width=True)

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
    except Exception as e:
        print(f'Error:{e}')
        


if __name__ == "__main__":
    operate_trading_tool()