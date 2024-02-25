import streamlit as st 
from datetime import date

import yfinance as yf 
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go 

import requests
from bs4 import BeautifulSoup


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predictions")

response = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', {'id': 'constituents'})
stocks = []

for row in table.findAll('tr')[1:]:
  ticker = row.findAll('td')[0].text.strip()
  stocks.append(ticker)

stocks = tuple(stocks)
selected = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def loadData(ticker):
  data = yf.download(ticker, START, TODAY)
  data.reset_index(inplace=True)
  return data

data_loadState = st.text("Load data...")
data = loadData(selected)
data_loadState.text("Loading data...done!")

st.subheader("Raw data")
st.write(data.tail())

def plotData():
  figure = go.Figure()
  figure.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
  figure.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
  figure.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
  st.plotly_chart(figure)

plotData()


#PREDICTIONS/FORECASTING
df_training = data[['Date', 'Close']]
df_training = df_training.rename(columns={"Date":"ds", "Close":"y"})

model = Prophet()
model.fit(df_training)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader("Forecasted data")
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = model.plot_components(forecast)
st.write(fig2)