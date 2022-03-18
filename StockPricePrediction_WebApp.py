# make sure you pip install streamlit fbprophet yfinance plotly

import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

#Layout
st.title('Stock Prediction App')

st.sidebar.header('About')
st.sidebar.markdown("""
* This app predicts the prices of selected tickers, the forecasts are based on the yahoo finance's stock market datas from 2010 till now
* **Python libraries:** plotly, streamlit, yfinance, fbprophet, datetime
* **Data source:** [YahooFinance](http://finance.yahoo.com).
""")

#Setup
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ('GOOG', 'AAPL', 'AMZN', 'MSFT', 'GME')
selected_stock = st.selectbox('Select a ticker for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 3)
period = n_years * 365

#Getting data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading...')
data = load_data(selected_stock)
data_load_state.text('Loading Done!')

st.subheader('Data')
st.write(data.tail())

# Ploting data
def plot_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_data()

# Prediction with Prophet
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot predictions
st.subheader('Predictions')
st.write(forecast.tail())

st.write(f'Prediction plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Prediction details")
fig2 = m.plot_components(forecast)
st.write(fig2)
