import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / "kronos"))

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm

from config import (
    LOOKBACK, PRED_LEN, REBALANCE_EVERY, INITIAL_CAPITAL,
    TOP_K, TEMPERATURE, TOP_P, SAMPLE_COUNT,
)


@dataclass
class BacktestResult:
    equity_curve: pd.Series          # portfolio value at each rebalance step
    benchmark_curve: pd.Series       # buy-and-hold benchmark value
    trade_log: list[dict] = field(default_factory=list)
    prediction_log: list[dict] = field(default_factory=list)


def _build_predictor(model_name: str, tokenizer_name: str, device: str):
    from model import Kronos, KronosTokenizer, KronosPredictor
    print(f"Loading tokenizer from {tokenizer_name} ...")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    print(f"Loading model from {model_name} ...")
    model = Kronos.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return KronosPredictor(model, tokenizer, max_context=LOOKBACK)


def _predict_return(predictor, df_slice: pd.DataFrame, x_ts: pd.Series, y_ts: pd.Series) -> float:
    """Run Kronos on a window, return predicted % return over pred_len candles."""
    try:
        pred_df = predictor.predict(
            df=df_slice[["open", "high", "low", "close", "volume"]],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=PRED_LEN,
            T=TEMPERATURE,
            top_p=TOP_P,
            sample_count=SAMPLE_COUNT,
            verbose=False,
        )
        current_close = df_slice["close"].iloc[-1]
        predicted_close = pred_df["close"].iloc[-1]
        return float((predicted_close - current_close) / current_close)
    except Exception as e:
        print(f"    Prediction error: {e}")
        return 0.0


def run_backtest(
    ticker_data: dict[str, pd.DataFrame],
    model_name: str,
    tokenizer_name: str,
    device: str = "cpu",
    benchmark_ticker: str = "SPY",
) -> BacktestResult:
    """
    Rolling-window backtest:
      Every REBALANCE_EVERY candles, generate signals for all tickers,
      rank by predicted return, go long top-K tickers equally weighted.
    """
    predictor = _build_predictor(model_name, tokenizer_name, device)

    # Align all tickers to a common date index
    all_dates = sorted(set.intersection(*[set(df.index) for df in ticker_data.values()]))
    all_dates = pd.DatetimeIndex(all_dates)

    # Need at least LOOKBACK + PRED_LEN candles to start
    start_idx = LOOKBACK

    capital = INITIAL_CAPITAL
    equity_points = {}
    trade_log = []
    prediction_log = []

    # Benchmark: buy-and-hold the benchmark ticker from the start
    bm_data = ticker_data.get(benchmark_ticker)
    bm_start_price = bm_data.loc[all_dates[start_idx], "close"] if bm_data is not None else None
    bm_shares = (capital / bm_start_price) if bm_start_price else 0

    current_positions: dict[str, float] = {}  # ticker -> shares held
    entry_prices: dict[str, float] = {}

    rebalance_steps = range(start_idx, len(all_dates) - PRED_LEN, REBALANCE_EVERY)

    for step in tqdm(rebalance_steps, desc="Backtesting"):
        date = all_dates[step]

        # --- Generate signals ---
        signals: dict[str, float] = {}
        for ticker, df in ticker_data.items():
            if ticker == benchmark_ticker:
                continue
            available = df.index[df.index <= date]
            if len(available) < LOOKBACK:
                continue
            window = df.loc[available[-LOOKBACK:]]
            x_ts = pd.Series(window.index)
            future_end = step + PRED_LEN
            if future_end >= len(all_dates):
                continue
            future_dates = all_dates[step + 1: step + PRED_LEN + 1]
            y_ts = pd.Series(future_dates)
            pred_return = _predict_return(predictor, window.reset_index(), x_ts, y_ts)
            signals[ticker] = pred_return
            prediction_log.append({"date": date, "ticker": ticker, "predicted_return": pred_return})

        if not signals:
            equity_points[date] = capital
            continue

        # --- Liquidate existing positions ---
        for ticker, shares in current_positions.items():
            if ticker in ticker_data:
                exit_price = ticker_data[ticker].loc[date, "close"] if date in ticker_data[ticker].index else entry_prices.get(ticker, 0)
                proceeds = shares * exit_price
                pnl = proceeds - shares * entry_prices.get(ticker, exit_price)
                capital += proceeds
                trade_log.append({
                    "date": date, "ticker": ticker, "action": "sell",
                    "shares": shares, "price": exit_price, "pnl": pnl,
                })
        current_positions = {}
        entry_prices = {}

        # --- Buy top-K ---
        ranked = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        top_picks = [t for t, _ in ranked[:TOP_K] if _ > 0]  # only buy if positive signal

        if top_picks:
            alloc_per = capital / len(top_picks)
            for ticker in top_picks:
                price = ticker_data[ticker].loc[date, "close"] if date in ticker_data[ticker].index else None
                if price is None or price <= 0:
                    continue
                shares = alloc_per / price
                current_positions[ticker] = shares
                entry_prices[ticker] = price
                capital -= alloc_per
                trade_log.append({
                    "date": date, "ticker": ticker, "action": "buy",
                    "shares": shares, "price": price, "pnl": 0,
                })

        # Portfolio value = cash + mark-to-market positions
        portfolio_value = capital + sum(
            current_positions.get(t, 0) * ticker_data[t].loc[date, "close"]
            for t in current_positions
            if date in ticker_data[t].index
        )
        equity_points[date] = portfolio_value

    equity_curve = pd.Series(equity_points, name="portfolio")
    equity_curve.index = pd.to_datetime(equity_curve.index)

    # Benchmark curve
    bm_curve = pd.Series(dtype=float, name="benchmark")
    if bm_data is not None and bm_shares > 0:
        bm_prices = bm_data.loc[bm_data.index.isin(equity_curve.index), "close"]
        bm_curve = (bm_prices * bm_shares).rename("benchmark")

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=bm_curve,
        trade_log=trade_log,
        prediction_log=prediction_log,
    )
