# 📈 SMA-RSI Trading Bot

A Python-based algorithmic trading bot that generates **BUY**, **SELL**, or **HOLD** signals using a dual Simple Moving Average (SMA) crossover strategy combined with RSI (Relative Strength Index) momentum filtering. Includes a built-in backtesting engine powered by [backtesting.py](https://kernc.github.io/backtesting.py/).

---

## Table of Contents

- [How It Works](#how-it-works)
  - [Strategy Logic](#strategy-logic)
  - [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Bot](#running-the-bot)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Example Output](#example-output)
- [What Can Be Improved](#what-can-be-improved)
- [Disclaimer](#disclaimer)

---

## How It Works

### Strategy Logic

The bot uses a **SMA Crossover + RSI Filter** strategy — a classic combination of trend-following and momentum analysis:

```
BUY  when:  Fast SMA crosses ABOVE Slow SMA  AND  RSI < 35 (oversold)
SELL when:  Fast SMA crosses BELOW Slow SMA  AND  RSI > 65 (overbought)
HOLD when:  Conditions for BUY or SELL are not both met
```

| Indicator | Default | Purpose |
|-----------|---------|---------|
| Fast SMA | 20-period | Tracks short-term price trend |
| Slow SMA | 50-period | Tracks long-term price trend |
| RSI | 14-period | Measures momentum (0–100 scale) |
| RSI Oversold | < 35 | Confirms buying opportunity |
| RSI Overbought | > 65 | Confirms selling opportunity |

**Why both conditions?** Using SMA crossover alone produces many false signals in choppy markets. The RSI filter adds a momentum confirmation layer — the bot only enters when the trend *and* momentum agree.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        bot.py                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  fetch_market_data()                                     │
│    │  Downloads OHLCV candles via yfinance               │
│    │  Falls back to synthetic data if unavailable        │
│    ▼                                                     │
│  build_signal()                                          │
│    │  Computes SMA(20), SMA(50), RSI(14)                 │
│    │  Evaluates crossover + momentum conditions          │
│    │  Returns BUY / SELL / HOLD with reasoning           │
│    ▼                                                     │
│  run_backtest()                                          │
│    │  Feeds data into backtesting.py engine              │
│    │  Simulates trades with stop-loss & take-profit      │
│    ▼                                                     │
│  Performance Summary                                     │
│    Return %, Sharpe Ratio, Max Drawdown, # Trades, etc.  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Key components:**

- **`fetch_market_data()`** — Downloads historical OHLCV data from Yahoo Finance. If `yfinance` is unavailable or the download fails, it generates synthetic price data for testing.
- **`compute_rsi()`** — Implements the standard Wilder's RSI calculation using exponential smoothing of gains and losses.
- **`build_signal()`** — Evaluates the latest market state and returns a signal dict with the decision, price, indicator values, and human-readable reasoning.
- **`run_backtest()`** — Runs a full historical simulation using `backtesting.py`, applying stop-loss (2%) and take-profit (5%) on each trade. Falls back to signal-only mode if the backtesting library isn't installed.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Internet connection (for fetching live market data)

### Installation

```bash
# Clone or navigate to the project
cd "trading bot"

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# Install dependencies
pip install pandas numpy yfinance backtesting
```

**Optional dependencies:**

```bash
pip install TA-Lib    # Uses optimized C library for indicators (requires system-level TA-Lib)
```

### Running the Bot

```bash
source venv/bin/activate
python3 bot.py
```

---

## Configuration

You can tune the strategy by modifying the parameters passed to `build_signal()` or the class attributes in `_Strategy`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | `"AAPL"` | Ticker symbol to fetch (passed to `fetch_market_data`) |
| `start` / `end` | `"2022-01-01"` / `"2026-01-01"` | Date range for historical data |
| `fast_sma_period` | `20` | Lookback for the fast moving average |
| `slow_sma_period` | `50` | Lookback for the slow moving average |
| `rsi_period` | `14` | Lookback for RSI calculation |
| `rsi_oversold` | `35` | RSI threshold to confirm BUY signals |
| `rsi_overbought` | `65` | RSI threshold to confirm SELL signals |
| `cash` | `10,000` | Starting capital for backtesting |
| `commission` | `0.1%` | Per-trade commission rate |

**Example — relaxing thresholds to generate more trades:**

```python
signal = build_signal(data, rsi_oversold=45, rsi_overbought=55)
```

---

## Project Structure

```
trading bot/
├── bot.py          # Main script: data fetching, indicators, signal generation, backtesting
├── venv/           # Python virtual environment
└── README.md       # This file
```

---

## Example Output

```
Fetching market data...
Loaded 1003 candles for the current session.
Current signal: HOLD | price: $270.92 | RSI: 45.69
Reason: Trend and momentum are not aligned with a clear trade setup.

========================================
BACKTEST PERFORMANCE SUMMARY
========================================
Start                     2022-01-03
End                       2025-12-31
Duration                  1458 days
Equity Final [$]          10000.0
Buy & Hold Return [%]    78.74
# Trades                  0
```

> **Note:** 0 trades means the strategy's entry conditions (SMA crossover *and* RSI < 35) were never simultaneously met for the given ticker and time range. This is expected for strong uptrend stocks like AAPL. Try different tickers or relaxed thresholds.

---

## What Can Be Improved

### 🔴 Strategy Improvements

| Area | Current State | Suggested Improvement |
|------|--------------|----------------------|
| **Entry conditions too strict** | Requires SMA crossover AND RSI confirmation simultaneously | Decouple conditions: allow SMA-only or RSI-only entries with smaller position sizes |
| **No position sizing** | All-in on every trade | Add risk-based position sizing (e.g., risk 1-2% of capital per trade) |
| **Fixed stop-loss / take-profit** | Hardcoded at 2% SL / 5% TP | Use ATR-based dynamic stops that adapt to volatility |
| **Single timeframe** | Only analyzes one timeframe | Add multi-timeframe confirmation (e.g., daily trend + 4H entry) |
| **No trailing stop** | Profits aren't protected as price moves | Implement trailing stop-loss to lock in gains during strong moves |

### 🟡 Technical Improvements

| Area | Current State | Suggested Improvement |
|------|--------------|----------------------|
| **Single ticker** | Only analyzes one symbol at a time | Support scanning a watchlist of tickers and ranking by signal strength |
| **No persistence** | Results are only printed to console | Save signals and backtest results to CSV/database |
| **No live trading** | Signal-only, no order execution | Integrate with a broker API (Alpaca, IBKR, Binance) for paper/live trading |
| **No alerts** | Must manually run the script | Add Telegram/Discord/email notifications on BUY/SELL signals |
| **No scheduling** | Runs once and exits | Add a scheduler (cron / APScheduler) to run at market open/close |
| **Error handling** | Broad `except Exception: pass` in data fetching | Add specific exception handling and logging |
| **No tests** | No unit or integration tests | Add pytest suite covering RSI calculation, signal logic, edge cases |

### 🟢 Additional Indicators to Consider

- **MACD** — Trend strength and crossover confirmation
- **Bollinger Bands** — Volatility-based entries at band extremes
- **Volume Profile / OBV** — Confirm moves with volume analysis
- **ADX** — Filter out signals during low-trend (ranging) markets
- **EMA** — Replace SMA with Exponential Moving Averages for faster response

---

## Disclaimer

> **⚠️ This bot is for educational and research purposes only.** It is not financial advice. Algorithmic trading involves significant risk — you can lose some or all of your capital. Always backtest thoroughly and paper trade before risking real money. Past performance does not guarantee future results.
