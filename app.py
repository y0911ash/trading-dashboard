import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="AlgoTrader Pro — Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99,102,241,.08), rgba(99,102,241,.02));
    border: 1px solid rgba(99,102,241,.2);
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="stMetric"] label { font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.5px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 22px !important; }
.trade-buy { color: #10b981; font-weight: 600; }
.trade-sell { color: #ef4444; font-weight: 600; }
div[data-testid="stSidebar"] { background: rgba(17,24,39,.95); }
h1 { letter-spacing: -0.5px !important; }
</style>
""", unsafe_allow_html=True)

# ─── INDICATOR FUNCTIONS ───
def calc_sma(series, period):
    return series.rolling(window=period).mean()

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    # Use Wilder's smoothing after initial SMA
    for i in range(period + 1, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_bollinger(series, period=20, std_mult=2):
    mid = calc_sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ─── STRATEGY SIGNALS ───
def get_signals(strategy, df, params):
    closes = df['Close']
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0  # 1=BUY, -1=SELL

    if strategy == "Moving Average Crossover":
        fn = calc_ema if params.get('ma_type') == 'EMA' else calc_sma
        short_ma = fn(closes, params['short'])
        long_ma = fn(closes, params['long'])
        for i in range(1, len(closes)):
            if pd.isna(short_ma.iloc[i]) or pd.isna(long_ma.iloc[i]) or pd.isna(short_ma.iloc[i-1]) or pd.isna(long_ma.iloc[i-1]):
                continue
            if short_ma.iloc[i-1] <= long_ma.iloc[i-1] and short_ma.iloc[i] > long_ma.iloc[i]:
                signals.iloc[i, 0] = 1
            elif short_ma.iloc[i-1] >= long_ma.iloc[i-1] and short_ma.iloc[i] < long_ma.iloc[i]:
                signals.iloc[i, 0] = -1
        return signals, {'Short MA': short_ma, 'Long MA': long_ma}

    elif strategy == "RSI Momentum":
        rsi = calc_rsi(closes, params['period'])
        for i in range(1, len(closes)):
            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i-1]):
                continue
            if rsi.iloc[i-1] <= params['oversold'] and rsi.iloc[i] > params['oversold']:
                signals.iloc[i, 0] = 1
            elif rsi.iloc[i-1] >= params['overbought'] and rsi.iloc[i] < params['overbought']:
                signals.iloc[i, 0] = -1
        return signals, {'RSI': rsi}

    elif strategy == "Bollinger Bands":
        mid, upper, lower = calc_bollinger(closes, params['period'], params['std'])
        for i in range(1, len(closes)):
            if pd.isna(lower.iloc[i]):
                continue
            if closes.iloc[i] <= lower.iloc[i]:
                signals.iloc[i, 0] = 1
            elif closes.iloc[i] >= upper.iloc[i]:
                signals.iloc[i, 0] = -1
        return signals, {'BB Mid': mid, 'BB Upper': upper, 'BB Lower': lower}

    elif strategy == "MACD":
        macd_line, signal_line, histogram = calc_macd(closes, params['fast'], params['slow'], params['signal'])
        for i in range(1, len(closes)):
            if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]) or pd.isna(macd_line.iloc[i-1]) or pd.isna(signal_line.iloc[i-1]):
                continue
            if macd_line.iloc[i-1] <= signal_line.iloc[i-1] and macd_line.iloc[i] > signal_line.iloc[i]:
                signals.iloc[i, 0] = 1
            elif macd_line.iloc[i-1] >= signal_line.iloc[i-1] and macd_line.iloc[i] < signal_line.iloc[i]:
                signals.iloc[i, 0] = -1
        return signals, {'MACD': macd_line, 'Signal': signal_line, 'Histogram': histogram}

    return signals, {}

# ─── BACKTEST ENGINE ───
def run_backtest(df, signals, capital, commission):
    cash = capital
    shares = 0
    position = None
    trades = []
    equity = []
    buy_hold = []

    # Filter to only actual signals (long-only, one position at a time)
    in_position = False
    filtered = []
    for i in range(len(signals)):
        sig = signals['signal'].iloc[i]
        if sig == 1 and not in_position:
            filtered.append((i, 'BUY'))
            in_position = True
        elif sig == -1 and in_position:
            filtered.append((i, 'SELL'))
            in_position = False

    sig_map = {}
    for idx, typ in filtered:
        sig_map[idx] = typ

    eq_cash = capital
    eq_shares = 0
    cum_return = 0.0

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]

        if sig_map.get(i) == 'BUY' and eq_shares == 0:
            cost = price * (1 + commission) if commission else price
            eq_shares = int(eq_cash // cost)
            if eq_shares > 0:
                eq_cash -= eq_shares * cost
                buy_price = price
                trades.append({
                    'Date': date.strftime('%Y-%m-%d'), 'Type': 'BUY',
                    'Price': round(price, 2), 'Shares': eq_shares,
                    'P&L': 0.0, 'Cum Return': round(cum_return, 2)
                })
        elif sig_map.get(i) == 'SELL' and eq_shares > 0:
            proceeds = price * (1 - commission) if commission else price
            pnl = (proceeds - buy_price) * eq_shares
            eq_cash += eq_shares * proceeds
            cum_return = ((eq_cash - capital) / capital) * 100
            trades.append({
                'Date': date.strftime('%Y-%m-%d'), 'Type': 'SELL',
                'Price': round(price, 2), 'Shares': eq_shares,
                'P&L': round(pnl, 2), 'Cum Return': round(cum_return, 2)
            })
            eq_shares = 0

        equity.append(eq_cash + eq_shares * price)
        buy_hold.append(capital * (price / df['Close'].iloc[0]))

    equity = np.array(equity)
    buy_hold = np.array(buy_hold)

    # Metrics
    daily_returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
    avg_r = np.mean(daily_returns) if len(daily_returns) else 0
    std_r = np.std(daily_returns) if len(daily_returns) else 1
    sharpe = (avg_r / std_r) * np.sqrt(252) if std_r > 0 else 0
    neg_returns = daily_returns[daily_returns < 0]
    down_dev = np.sqrt(np.mean(neg_returns**2)) if len(neg_returns) > 0 else 1
    sortino = (avg_r / down_dev) * np.sqrt(252) if down_dev > 0 else 0
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    max_dd = np.max(drawdown) if len(drawdown) else 0
    total_return = ((equity[-1] - capital) / capital) * 100 if len(equity) else 0
    sell_trades = [t for t in trades if t['Type'] == 'SELL']
    wins = len([t for t in sell_trades if t['P&L'] > 0])
    win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
    avg_profit = (sum(t['P&L'] for t in sell_trades) / len(sell_trades)) if sell_trades else 0

    metrics = {
        'Total Return': total_return, 'Sharpe Ratio': sharpe, 'Sortino Ratio': sortino,
        'Max Drawdown': max_dd, 'Win Rate': win_rate,
        'Total Trades': len(sell_trades), 'Avg Profit/Trade': avg_profit
    }

    return trades, equity, buy_hold, metrics, sig_map

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("### 📈 AlgoTrader Pro")
    st.caption("Strategy Backtester")
    st.divider()

    st.markdown("**MARKET DATA**")
    ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="e.g. AAPL, TSLA, MSFT").upper().strip()
    range_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}
    date_range = st.select_slider("Date Range", options=list(range_map.keys()), value="1Y")
    st.divider()

    st.markdown("**STRATEGY**")
    strategy = st.selectbox("Select Strategy", [
        "Moving Average Crossover", "RSI Momentum", "Bollinger Bands", "MACD"
    ])

    params = {}
    if strategy == "Moving Average Crossover":
        params['ma_type'] = st.selectbox("MA Type", ["SMA", "EMA"])
        c1, c2 = st.columns(2)
        params['short'] = c1.number_input("Short Window", value=20, min_value=2, max_value=100)
        params['long'] = c2.number_input("Long Window", value=50, min_value=5, max_value=200)
    elif strategy == "RSI Momentum":
        params['period'] = st.number_input("RSI Period", value=14, min_value=2, max_value=50)
        c1, c2 = st.columns(2)
        params['oversold'] = c1.number_input("Oversold", value=30, min_value=5, max_value=45)
        params['overbought'] = c2.number_input("Overbought", value=70, min_value=55, max_value=95)
    elif strategy == "Bollinger Bands":
        c1, c2 = st.columns(2)
        params['period'] = c1.number_input("Period", value=20, min_value=5, max_value=50)
        params['std'] = c2.number_input("Std Dev", value=2.0, min_value=0.5, max_value=4.0, step=0.5)
    elif strategy == "MACD":
        c1, c2 = st.columns(2)
        params['fast'] = c1.number_input("Fast", value=12, min_value=2, max_value=50)
        params['slow'] = c2.number_input("Slow", value=26, min_value=5, max_value=100)
        params['signal'] = st.number_input("Signal", value=9, min_value=2, max_value=30)
    st.divider()

    st.markdown("**SETTINGS**")
    use_commission = st.toggle("Commission (0.1%)", value=False)
    capital = st.number_input("Starting Capital ($)", value=10000, min_value=1000, step=1000)
    st.divider()

    run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

# ─── MAIN ───
st.markdown("# 📊 AlgoTrader Pro")

if run_btn or 'results' in st.session_state:
    if run_btn:
        with st.spinner("Fetching data & running backtest..."):
            try:
                df = yf.download(ticker, period=range_map[date_range], interval="1d", progress=False)
                if df.empty:
                    st.error(f"No data found for ticker **{ticker}**. Please try another.")
                    st.stop()
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.stop()

            commission = 0.001 if use_commission else 0
            signals, overlays = get_signals(strategy, df, params)
            trades, equity, buy_hold, metrics, sig_map = run_backtest(df, signals, capital, commission)

            st.session_state['results'] = {
                'df': df, 'trades': trades, 'equity': equity, 'buy_hold': buy_hold,
                'metrics': metrics, 'sig_map': sig_map, 'overlays': overlays,
                'ticker': ticker, 'strategy': strategy
            }

    r = st.session_state['results']
    df, trades, equity, buy_hold, metrics = r['df'], r['trades'], r['equity'], r['buy_hold'], r['metrics']
    sig_map, overlays, ticker_name = r['sig_map'], r['overlays'], r['ticker']

    # ─── METRICS ───
    cols = st.columns(7)
    metric_items = [
        ("Total Return", f"{metrics['Total Return']:.2f}%"),
        ("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}"),
        ("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}"),
        ("Max Drawdown", f"{metrics['Max Drawdown']:.2f}%"),
        ("Win Rate", f"{metrics['Win Rate']:.1f}%"),
        ("Total Trades", f"{metrics['Total Trades']}"),
        ("Avg Profit/Trade", f"${metrics['Avg Profit/Trade']:.2f}"),
    ]
    for col, (label, value) in zip(cols, metric_items):
        col.metric(label, value)

    # ─── PRICE CHART ───
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name='Close', line=dict(color='#6366f1', width=1.5),
        fill='tozeroy', fillcolor='rgba(99,102,241,0.05)'
    ))

    overlay_colors = {'Short MA': '#f59e0b', 'Long MA': '#3b82f6', 'BB Mid': '#8b5cf6',
                      'BB Upper': '#ef4444', 'BB Lower': '#10b981'}
    for name, series in overlays.items():
        if name in ('RSI', 'MACD', 'Signal', 'Histogram'):
            continue
        dash = 'dash' if 'BB' in name and name != 'BB Mid' else 'solid'
        fig.add_trace(go.Scatter(
            x=df.index, y=series, name=name,
            line=dict(color=overlay_colors.get(name, '#888'), width=1.2, dash=dash)
        ))

    buy_dates = [df.index[i] for i in sig_map if sig_map[i] == 'BUY']
    buy_prices = [df['Close'].iloc[i] for i in sig_map if sig_map[i] == 'BUY']
    sell_dates = [df.index[i] for i in sig_map if sig_map[i] == 'SELL']
    sell_prices = [df['Close'].iloc[i] for i in sig_map if sig_map[i] == 'SELL']

    fig.add_trace(go.Scatter(
        x=buy_dates, y=buy_prices, mode='markers', name='BUY',
        marker=dict(symbol='triangle-up', size=14, color='#10b981', line=dict(width=1, color='white'))
    ))
    fig.add_trace(go.Scatter(
        x=sell_dates, y=sell_prices, mode='markers', name='SELL',
        marker=dict(symbol='triangle-down', size=14, color='#ef4444', line=dict(width=1, color='white'))
    ))

    fig.update_layout(
        title=f"📈 {ticker_name} — Price Chart & Signals",
        template='plotly_dark', height=500,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(42,48,66,0.4)'),
        yaxis=dict(gridcolor='rgba(42,48,66,0.4)', tickprefix='$'),
        legend=dict(orientation='h', y=1.12),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=60, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── EQUITY CURVE ───
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df.index, y=equity, name='Strategy',
        line=dict(color='#10b981', width=2), fill='tozeroy', fillcolor='rgba(16,185,129,0.05)'
    ))
    fig2.add_trace(go.Scatter(
        x=df.index, y=buy_hold, name='Buy & Hold',
        line=dict(color='#6366f1', width=1.5, dash='dash')
    ))
    fig2.update_layout(
        title="📊 Equity Curve — Strategy vs Buy & Hold",
        template='plotly_dark', height=350,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(42,48,66,0.4)'),
        yaxis=dict(gridcolor='rgba(42,48,66,0.4)', tickprefix='$'),
        legend=dict(orientation='h', y=1.12),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=60, b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ─── TRADE LOG ───
    st.markdown("### 📋 Trade Log")
    if trades:
        trade_df = pd.DataFrame(trades)
        def color_type(val):
            return 'color: #10b981; font-weight: 600' if val == 'BUY' else 'color: #ef4444; font-weight: 600'
        def color_pnl(val):
            return f'color: {"#10b981" if val >= 0 else "#ef4444"}'
        styled = trade_df.style.applymap(color_type, subset=['Type']).applymap(color_pnl, subset=['P&L', 'Cum Return'])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No trades generated for this strategy/period combination.")

else:
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;padding:80px 20px;'>"
        "<h3 style='color:#94a3b8;'>Welcome to AlgoTrader Pro</h3>"
        "<p style='color:#64748b;'>Enter a stock ticker, choose a strategy, and hit <b>Run Backtest</b> to get started.</p>"
        "</div>",
        unsafe_allow_html=True
    )
