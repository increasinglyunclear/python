# Breakouts01 by Kevin Walker
# Oct 2024
# First attempt at spotting breakouts
# This one using simple moving avg crossovers

# Import required libraries
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Input, Output

# Fetch stock data
def fetch_stock_data(symbol, period='1y'):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

# Detect breakouts
def detect_breakouts(df, ma_period=20):
    df['MA'] = df['Close'].rolling(window=ma_period).mean()
    df['Breakout'] = (df['Close'] > df['MA']) & (df['Close'].shift(1) <= df['MA'].shift(1))
    return df

# Draw candlestick chart
def create_chart(df):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    
    # Add Moving Average
    fig.add_trace(go.Scatter(x=df.index, y=df['MA'], name='MA'))
    
    # Add breakout points
    breakouts = df[df['Breakout']]
    fig.add_trace(go.Scatter(x=breakouts.index, y=breakouts['High'], 
                             mode='markers', name='Breakouts',
                             marker=dict(size=10, symbol='triangle-up', color='red')))
    
    fig.update_layout(title='Stock Price and Breakouts', xaxis_title='Date', yaxis_title='Price')
    return fig

# Initialize the Dash app
app = Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Stock Breakout Scanner'),
    dcc.Input(id='symbol-input', type='text', placeholder='Enter stock symbol'),
    dcc.Graph(id='stock-chart')
])

# Define callback to update chart
@app.callback(
    Output('stock-chart', 'figure'),
    Input('symbol-input', 'value')
)
def update_chart(symbol):
    if symbol:
        df = fetch_stock_data(symbol)
        df = detect_breakouts(df)
        return create_chart(df)
    return go.Figure()  # Return empty figure if no symbol entered

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
