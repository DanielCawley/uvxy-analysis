"""
UVXY Analysis Website
Fetches UVXY data from Yahoo Finance and provides various strategies, data, and statistics.
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
import time

app = Flask(__name__)

def fetch_uvxy_data():
    """Fetch UVXY data from Yahoo Finance using direct API (2021-2026)"""
    # Use Yahoo Finance's query1 API directly
    symbol = "UVXY"
    start_timestamp = int(datetime(2021, 1, 1).timestamp())
    end_timestamp = int(datetime(2026, 1, 16).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_timestamp,
        "period2": end_timestamp,
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
        raise Exception("Failed to fetch data from Yahoo Finance")

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "Date": pd.to_datetime(timestamps, unit='s'),
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
        "Volume": quote["volume"]
    })

    # Remove any rows with NaN values
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df

def calculate_statistics(df):
    """Calculate comprehensive statistics on UVXY data"""
    stats = {}

    # Basic Statistics
    stats['basic'] = {
        'current_price': round(df['Close'].iloc[-1], 2),
        'all_time_high': round(df['High'].max(), 2),
        'all_time_low': round(df['Low'].min(), 2),
        'average_price': round(df['Close'].mean(), 2),
        'median_price': round(df['Close'].median(), 2),
        'std_deviation': round(df['Close'].std(), 2),
        'total_trading_days': len(df),
        'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
    }

    # Returns Statistics
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)) * 100

    stats['returns'] = {
        'avg_daily_return': round(df['Daily_Return'].mean(), 4),
        'max_daily_gain': round(df['Daily_Return'].max(), 2),
        'max_daily_loss': round(df['Daily_Return'].min(), 2),
        'positive_days': int((df['Daily_Return'] > 0).sum()),
        'negative_days': int((df['Daily_Return'] < 0).sum()),
        'win_rate': round((df['Daily_Return'] > 0).sum() / len(df.dropna()) * 100, 2),
        'avg_gain_on_up_days': round(df[df['Daily_Return'] > 0]['Daily_Return'].mean(), 2),
        'avg_loss_on_down_days': round(df[df['Daily_Return'] < 0]['Daily_Return'].mean(), 2),
    }

    # Calculate total return over period
    total_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
    stats['returns']['total_return_pct'] = round(total_return, 2)

    # Volatility Statistics
    stats['volatility'] = {
        'daily_volatility': round(df['Daily_Return'].std(), 2),
        'annualized_volatility': round(df['Daily_Return'].std() * np.sqrt(252), 2),
        'avg_true_range': round((df['High'] - df['Low']).mean(), 2),
        'max_intraday_range': round((df['High'] - df['Low']).max(), 2),
        'avg_volume': int(df['Volume'].mean()),
        'max_volume': int(df['Volume'].max()),
    }

    # Drawdown Analysis
    df['Cumulative_Max'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Cumulative_Max']) / df['Cumulative_Max'] * 100

    stats['drawdown'] = {
        'max_drawdown': round(df['Drawdown'].min(), 2),
        'avg_drawdown': round(df['Drawdown'].mean(), 2),
        'current_drawdown': round(df['Drawdown'].iloc[-1], 2),
    }

    # Monthly Statistics
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_returns = df.groupby('Month')['Close'].agg(['first', 'last'])
    monthly_returns['return'] = (monthly_returns['last'] / monthly_returns['first'] - 1) * 100

    stats['monthly'] = {
        'best_month': round(monthly_returns['return'].max(), 2),
        'worst_month': round(monthly_returns['return'].min(), 2),
        'avg_monthly_return': round(monthly_returns['return'].mean(), 2),
        'positive_months': int((monthly_returns['return'] > 0).sum()),
        'negative_months': int((monthly_returns['return'] < 0).sum()),
    }

    # Yearly Statistics
    df['Year'] = df['Date'].dt.year
    yearly_stats = []
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        if len(year_data) > 1:
            year_return = ((year_data['Close'].iloc[-1] / year_data['Close'].iloc[0]) - 1) * 100
            yearly_stats.append({
                'year': int(year),
                'return': round(year_return, 2),
                'high': round(year_data['High'].max(), 2),
                'low': round(year_data['Low'].min(), 2),
                'avg_volume': int(year_data['Volume'].mean())
            })
    stats['yearly'] = yearly_stats

    return stats

def calculate_strategies(df):
    """Calculate various trading strategies and their performance"""
    strategies = {}

    # Calculate daily return first
    df['Daily_Return'] = df['Close'].pct_change() * 100

    # Strategy 1: Moving Average Crossover (10/50 SMA)
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['MA_Signal'] = np.where(df['SMA10'] > df['SMA50'], 1, -1)
    df['MA_Strategy_Return'] = df['MA_Signal'].shift(1) * df['Daily_Return']

    ma_total_return = df['MA_Strategy_Return'].sum()
    strategies['ma_crossover'] = {
        'name': 'Moving Average Crossover (10/50 SMA)',
        'description': 'Long when 10-day SMA > 50-day SMA, Short otherwise',
        'total_return': round(ma_total_return, 2),
        'avg_daily_return': round(df['MA_Strategy_Return'].mean(), 4),
        'win_rate': round((df['MA_Strategy_Return'] > 0).sum() / len(df.dropna()) * 100, 2),
        'sharpe_ratio': round(df['MA_Strategy_Return'].mean() / df['MA_Strategy_Return'].std() * np.sqrt(252), 2) if df['MA_Strategy_Return'].std() > 0 else 0
    }

    # Strategy 2: RSI Strategy
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought)
    df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    df['RSI_Strategy_Return'] = df['RSI_Signal'].shift(1) * df['Daily_Return']

    rsi_total_return = df['RSI_Strategy_Return'].sum()
    strategies['rsi'] = {
        'name': 'RSI Mean Reversion',
        'description': 'Long when RSI < 30 (oversold), Short when RSI > 70 (overbought)',
        'total_return': round(rsi_total_return, 2),
        'avg_daily_return': round(df['RSI_Strategy_Return'].mean(), 4),
        'win_rate': round((df['RSI_Strategy_Return'] > 0).sum() / max(len(df[df['RSI_Strategy_Return'] != 0]), 1) * 100, 2),
        'times_triggered': int((df['RSI_Signal'] != 0).sum())
    }

    # Strategy 3: Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    df['BB_Signal'] = np.where(df['Close'] < df['BB_Lower'], 1,
                               np.where(df['Close'] > df['BB_Upper'], -1, 0))
    df['BB_Strategy_Return'] = df['BB_Signal'].shift(1) * df['Daily_Return']

    bb_total_return = df['BB_Strategy_Return'].sum()
    strategies['bollinger'] = {
        'name': 'Bollinger Bands Mean Reversion',
        'description': 'Long when price below lower band, Short when above upper band',
        'total_return': round(bb_total_return, 2),
        'avg_daily_return': round(df['BB_Strategy_Return'].mean(), 4),
        'win_rate': round((df['BB_Strategy_Return'] > 0).sum() / max(len(df[df['BB_Strategy_Return'] != 0]), 1) * 100, 2),
        'times_triggered': int((df['BB_Signal'] != 0).sum())
    }

    # Strategy 4: Momentum Strategy
    df['Momentum_5'] = df['Close'].pct_change(5) * 100
    df['Mom_Signal'] = np.where(df['Momentum_5'] > 5, -1,  # Short after big spike
                                np.where(df['Momentum_5'] < -5, 1, 0))  # Long after big drop
    df['Mom_Strategy_Return'] = df['Mom_Signal'].shift(1) * df['Daily_Return']

    mom_total_return = df['Mom_Strategy_Return'].sum()
    strategies['momentum'] = {
        'name': '5-Day Momentum Reversal',
        'description': 'Short after 5%+ gain, Long after 5%+ loss (mean reversion)',
        'total_return': round(mom_total_return, 2),
        'avg_daily_return': round(df['Mom_Strategy_Return'].mean(), 4),
        'win_rate': round((df['Mom_Strategy_Return'] > 0).sum() / max(len(df[df['Mom_Strategy_Return'] != 0]), 1) * 100, 2),
        'times_triggered': int((df['Mom_Signal'] != 0).sum())
    }

    # Strategy 5: Volume Spike Strategy
    df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Avg_Volume']
    df['Vol_Signal'] = np.where((df['Volume_Ratio'] > 2) & (df['Daily_Return'] > 0), -1,
                                np.where((df['Volume_Ratio'] > 2) & (df['Daily_Return'] < 0), 1, 0))
    df['Vol_Strategy_Return'] = df['Vol_Signal'].shift(1) * df['Daily_Return']

    vol_total_return = df['Vol_Strategy_Return'].sum()
    strategies['volume_spike'] = {
        'name': 'Volume Spike Reversal',
        'description': 'Fade moves on 2x average volume (expect mean reversion)',
        'total_return': round(vol_total_return, 2),
        'avg_daily_return': round(df['Vol_Strategy_Return'].mean(), 4),
        'win_rate': round((df['Vol_Strategy_Return'] > 0).sum() / max(len(df[df['Vol_Strategy_Return'] != 0]), 1) * 100, 2),
        'times_triggered': int((df['Vol_Signal'] != 0).sum())
    }

    # Strategy 6: Buy and Hold (Benchmark)
    buy_hold_return = df['Daily_Return'].sum()
    strategies['buy_hold'] = {
        'name': 'Buy and Hold (Benchmark)',
        'description': 'Simply hold UVXY throughout the period',
        'total_return': round(buy_hold_return, 2),
        'avg_daily_return': round(df['Daily_Return'].mean(), 4),
        'win_rate': round((df['Daily_Return'] > 0).sum() / len(df.dropna()) * 100, 2),
        'sharpe_ratio': round(df['Daily_Return'].mean() / df['Daily_Return'].std() * np.sqrt(252), 2) if df['Daily_Return'].std() > 0 else 0
    }

    # Strategy 7: Short Only (Given UVXY decay)
    df['Short_Return'] = -df['Daily_Return']
    short_total_return = df['Short_Return'].sum()
    strategies['short_only'] = {
        'name': 'Short Only Strategy',
        'description': 'Always short UVXY (exploit contango decay)',
        'total_return': round(short_total_return, 2),
        'avg_daily_return': round(df['Short_Return'].mean(), 4),
        'win_rate': round((df['Short_Return'] > 0).sum() / len(df.dropna()) * 100, 2),
        'sharpe_ratio': round(df['Short_Return'].mean() / df['Short_Return'].std() * np.sqrt(252), 2) if df['Short_Return'].std() > 0 else 0
    }

    return strategies

def get_chart_data(df):
    """Prepare data for charts"""
    # Price chart data
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    price_data = {
        'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'close': df['Close'].round(2).tolist(),
        'high': df['High'].round(2).tolist(),
        'low': df['Low'].round(2).tolist(),
        'open': df['Open'].round(2).tolist(),
        'volume': df['Volume'].tolist(),
        'sma10': df['SMA10'].round(2).fillna(0).tolist(),
        'sma50': df['SMA50'].round(2).fillna(0).tolist(),
    }

    # Calculate RSI for chart
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    price_data['rsi'] = df['RSI'].round(2).fillna(50).tolist()

    # Daily returns histogram data
    df['Daily_Return'] = df['Close'].pct_change() * 100
    returns = df['Daily_Return'].dropna().tolist()

    # Monthly returns for heatmap
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    monthly_pivot = df.pivot_table(values='Daily_Return', index='Year', columns='Month', aggfunc='sum')
    monthly_data = monthly_pivot.round(2).fillna(0).to_dict('index')

    return {
        'price': price_data,
        'returns': returns,
        'monthly': monthly_data
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    try:
        df = fetch_uvxy_data()
        statistics = calculate_statistics(df.copy())
        strategies = calculate_strategies(df.copy())
        chart_data = get_chart_data(df.copy())

        return jsonify({
            'success': True,
            'statistics': statistics,
            'strategies': strategies,
            'charts': chart_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
