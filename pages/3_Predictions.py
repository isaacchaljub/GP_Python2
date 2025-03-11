import polars as pl
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_data import COM, PRI, FinancialData


###########################################################

st.set_page_config(layout='wide', page_title="Predicted financial data", page_icon="📈")

#st.markdown("# Plotting Page")
st.sidebar.header("Predictions Page")


# @st.cache_data
def plot_predicted_data():
    comps = st.sidebar.selectbox("Select Company", COM['Company Name'].to_list())
    tk=COM.filter(pl.col('Company Name')==comps)['Ticker'].to_list()
    #Pri=PRI.with_columns(pl.col('Date').str.to_datetime('%Y-%m-%d').cast(pl.Date))

    min_date=PRI.filter(pl.col('Ticker').is_in(tk))['Date'].min()
    max_date=PRI.filter(pl.col('Ticker').is_in(tk))['Date'].max()

    start= st.sidebar.date_input(label='start date', key='start_date')
    #start= st.sidebar.time_input(label='start date',value=None)

    end= st.sidebar.date_input(label='start date', key='end_date')
    #end= st.sidebar.time_input(label='end date',value=None)

    start_str = start.strftime('%Y-%m-%d')
    end_str   = end.strftime('%Y-%m-%d')
    #DESTINATION = st.sidebar.selectbox("Select Destination", destinations)
    

    #print(tk)
    fp=FinancialData(tk)
    data, preds= fp.predictions(start_str, end_str)


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
            y=preds["returns"],
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


if __name__ == "__main__":
    # This is to configure some aspects of the app
    # st.set_page_config(
    #     layout="wide", page_title="Financial Market Analysis and Prediction Tool", page_icon="📈"
    # )

    # Write titles in the main frame and the side bar
    st.title("Actual vs predicted Stock Returns")
    #st.sidebar.title("Select a Company")

    plot_predicted_data()