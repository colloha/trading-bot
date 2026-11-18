import math
import pandas as pd
from typing import Iterable

import numpy as np

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency
    yf = None

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
except ImportError:  # pragma: no cover - optional dependency
    Backtest = None
    Strategy = object
    crossover = None

try:
    import talib
except ImportError:  # pragma: no cover - optional dependency
    talib = None

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _as_rows(data):
    """Normalize OHLCV inputs into a list of row dictionaries."""
    if data is None:
        return []

    rows = []
    if hasattr(data, "to_dict"):
        rows = data.to_dict("records")
    elif isinstance(data, dict):
        keys = list(data.keys())
        count = len(next(iter(data.values()), [])) if keys else 0
        for idx in range(count):
            rows.append({key: data[key][idx] for key in keys})
    elif isinstance(data, Iterable):
        rows = list(data)
    else:
        raise TypeError("Unsupported OHLCV data format.")

    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cleaned = {}
        for key, value in row.items():
            cleaned[str(key).strip()] = value
        cleaned = {"Open": cleaned.get("Open", cleaned.get("open")), "High": cleaned.get("High", cleaned.get("high")), "Low": cleaned.get("Low", cleaned.get("low")), "Close": cleaned.get("Close", cleaned.get("close")), "Volume": cleaned.get("Volume", cleaned.get("volume"))}
        normalized.append(cleaned)
    return normalized


def fetch_market_data(symbol="AAPL", start="2022-01-01", end="2026-01-01"):
    """Return OHLCV data as a pandas DataFrame. Uses Yahoo Finance when available and falls back to synthetic data."""
    if yf is not None:
        try:
            data = yf.download(symbol, start=start, end=end, progress=False)
            if hasattr(data, "empty") and data.empty:
                raise ValueError(f"No market data available for {symbol}.")
            # yfinance ≥0.2.31 returns MultiIndex columns: (Price, Ticker).
            # Flatten them to plain "Open", "Close", etc.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            # Keep only the standard OHLCV columns and drop NaNs
            data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if not data.empty:
                return data
        except Exception:
            pass

    rng = np.random.default_rng(42)
    periods = 500
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=periods, name="Date")
    drift = np.cumsum(rng.normal(0, 1.5, periods))
    open_prices = 100 + drift
    close_prices = open_prices + rng.normal(0, 1.0, periods)
    high_prices = np.maximum(open_prices, close_prices) + rng.uniform(0.5, 2.5, periods)
    low_prices = np.minimum(open_prices, close_prices) - rng.uniform(0.5, 2.5, periods)
    volume = rng.integers(700_000, 2_000_000, periods)
    return pd.DataFrame({
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
        "Volume": volume.astype(float),
    }, index=dates)


def compute_rsi(series, period=14):
    """Compute a standard RSI from a numeric array."""
    series = np.asarray(series, dtype=float)
    if series.size <= period:
        return np.full_like(series, 50.0, dtype=float)

    deltas = np.diff(series, prepend=series[0])
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)

    avg_gain = np.zeros_like(series, dtype=float)
    avg_loss = np.zeros_like(series, dtype=float)
    avg_gain[period] = gains[1:period + 1].mean()
    avg_loss[period] = losses[1:period + 1].mean()

    for idx in range(period + 1, len(series)):
        avg_gain[idx] = ((avg_gain[idx - 1] * (period - 1)) + gains[idx]) / period
        avg_loss[idx] = ((avg_loss[idx - 1] * (period - 1)) + losses[idx]) / period

    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 1.0), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    return np.clip(rsi, 0, 100)


def build_signal(data, fast_sma_period=20, slow_sma_period=50, rsi_period=14, rsi_oversold=35, rsi_overbought=65):
    """Return a BUY, SELL, or HOLD signal based on SMA crossover and RSI momentum."""
    rows = _as_rows(data)
    if not rows:
        raise ValueError("No market data available to generate a signal.")

    close = np.asarray([float(row["Close"]) for row in rows if row.get("Close") is not None], dtype=float)
    if close.size < max(slow_sma_period, rsi_period):
        return {
            "signal": "HOLD",
            "price": float(close[-1]) if close.size else 0.0,
            "rsi": 50.0,
            "reason": "Insufficient history for a valid signal.",
        }

    sma_fast = np.convolve(close, np.ones(fast_sma_period) / fast_sma_period, mode="valid")
    sma_slow = np.convolve(close, np.ones(slow_sma_period) / slow_sma_period, mode="valid")
    rsi = compute_rsi(close, period=rsi_period)

    idx = -1
    prev_fast = sma_fast[-2] if sma_fast.size >= 2 else sma_fast[-1]
    prev_slow = sma_slow[-2] if sma_slow.size >= 2 else sma_slow[-1]
    curr_fast = sma_fast[idx]
    curr_slow = sma_slow[idx]
    curr_rsi = rsi[idx]

    if prev_fast < prev_slow and curr_fast > curr_slow and curr_rsi < rsi_oversold:
        signal = "BUY"
        reason = "Fast SMA crossed above slow SMA and RSI is below oversold threshold."
    elif prev_fast > prev_slow and curr_fast < curr_slow and curr_rsi > rsi_overbought:
        signal = "SELL"
        reason = "Fast SMA crossed below slow SMA and RSI is above overbought threshold."
    else:
        signal = "HOLD"
        reason = "Trend and momentum are not aligned with a clear trade setup."

    return {
        "signal": signal,
        "price": float(close[-1]),
        "sma_fast": float(curr_fast),
        "sma_slow": float(curr_slow),
        "rsi": float(curr_rsi),
        "reason": reason,
    }


class SmaRsiStrategy:
    """A simple strategy object used when Backtesting is unavailable."""

    def __init__(self, data):
        self.data = _as_rows(data)

    def evaluate(self):
        return build_signal(self.data)


def run_backtest(data, cash=10_000, commission=0.001):
    """Run a strategy when the optional backtesting package is installed; otherwise return a lightweight summary."""
    if Backtest is None:
        signal = build_signal(data)
        return {
            "signal": signal["signal"],
            "price": signal["price"],
            "summary": "Backtesting library unavailable; signal-only mode is active.",
            "cash": cash,
            "commission": commission,
        }

    # Ensure data is a pandas DataFrame with proper OHLCV columns
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(_as_rows(data))

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = data[col].astype(float)

    class _Strategy(Strategy):
        fast_sma_period = 20
        slow_sma_period = 50
        rsi_period = 14
        rsi_oversold = 35
        rsi_overbought = 65

        def init(self):
            close = self.data.Close
            if talib is not None:
                self.sma_fast = self.I(talib.SMA, close, timeperiod=self.fast_sma_period)
                self.sma_slow = self.I(talib.SMA, close, timeperiod=self.slow_sma_period)
                self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
            else:
                self.sma_fast = self.I(lambda x: np.convolve(x, np.ones(self.fast_sma_period) / self.fast_sma_period, mode="valid"), close)
                self.sma_slow = self.I(lambda x: np.convolve(x, np.ones(self.slow_sma_period) / self.slow_sma_period, mode="valid"), close)
                self.rsi = self.I(lambda x: compute_rsi(x, self.rsi_period), close)

        def next(self):
            if len(self.data) < max(self.slow_sma_period, self.rsi_period):
                return
            if not self.position:
                if crossover(self.sma_fast, self.sma_slow) and self.rsi[-1] < self.rsi_oversold:
                    entry_price = self.data.Close[-1]
                    self.buy(sl=entry_price * 0.98, tp=entry_price * 1.05)
            elif crossover(self.sma_slow, self.sma_fast) or self.rsi[-1] > self.rsi_overbought:
                self.position.close()

    bt = Backtest(data, _Strategy, cash=cash, commission=commission, exclusive_orders=True)
    return bt.run()


def main():
    print("Fetching market data...")
    data = fetch_market_data()
    print(f"Loaded {len(data)} candles for the current session.")

    signal = build_signal(data)
    print(f"Current signal: {signal['signal']} | price: ${signal['price']:.2f} | RSI: {signal['rsi']:.2f}")
    print(f"Reason: {signal['reason']}")

    stats = run_backtest(data)
    print("\n" + "=" * 40)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("=" * 40)
    print(stats)


if __name__ == "__main__":
    main()