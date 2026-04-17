import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / "kronos"))

import json
from datetime import date
import pandas as pd
import numpy as np

from config import (
    TICKERS, BENCHMARK_TICKER, LOOKBACK, PRED_LEN,
    MODEL_NAME, TOKENIZER_NAME, DEVICE, TEMPERATURE, TOP_P,
    SAMPLE_COUNT, TOP_K, INITIAL_CAPITAL, PORTFOLIO_FILE,
)
from data.fetcher import fetch_ohlcv


def _load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "cash": INITIAL_CAPITAL,
        "positions": {},    # ticker -> {"shares": float, "entry_price": float}
        "history": [],      # list of {"date", "value"} snapshots
        "trade_log": [],
    }


def _save_portfolio(portfolio: dict) -> None:
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)


def _build_predictor():
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return KronosPredictor(model, tokenizer, max_context=LOOKBACK)


def run_paper_trade(tickers: list[str] | None = None) -> None:
    """
    Run one paper-trading cycle:
      1. Download latest data for each ticker
      2. Generate Kronos predictions
      3. Rank and print recommended trades
      4. Update virtual portfolio JSON
    """
    tickers = tickers or TICKERS
    today = str(date.today())

    print(f"\n{'='*50}")
    print(f"  Kronos Paper Trade — {today}")
    print(f"{'='*50}")

    # Fetch recent data (need at least LOOKBACK candles)
    print("\nFetching market data ...")
    ticker_data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = fetch_ohlcv(t, start="2020-01-01", end=today, force_refresh=True)
            if len(df) >= LOOKBACK:
                ticker_data[t] = df
        except Exception as e:
            print(f"  WARNING: {t}: {e}")

    if not ticker_data:
        print("No data available. Exiting.")
        return

    predictor = _build_predictor()
    signals: dict[str, float] = {}

    print("\nGenerating Kronos predictions ...")
    for ticker, df in ticker_data.items():
        window = df.iloc[-LOOKBACK:].reset_index()
        x_ts = window["timestamps"]
        # Fake future timestamps (business days) for y_ts
        last_date = pd.Timestamp(df.index[-1])
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=PRED_LEN)
        y_ts = pd.Series(future_dates)

        try:
            pred_df = predictor.predict(
                df=window[["open", "high", "low", "close", "volume"]],
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=PRED_LEN,
                T=TEMPERATURE,
                top_p=TOP_P,
                sample_count=SAMPLE_COUNT,
                verbose=False,
            )
            current_close = df["close"].iloc[-1]
            predicted_close = pred_df["close"].iloc[-1]
            predicted_return = (predicted_close - current_close) / current_close
            signals[ticker] = predicted_return
            direction = "UP" if predicted_return > 0 else "DOWN"
            print(f"  {ticker:>6}: {direction} {predicted_return:+.2%}  (current ${current_close:.2f} → pred ${predicted_close:.2f})")
        except Exception as e:
            print(f"  {ticker}: prediction failed — {e}")

    if not signals:
        print("No predictions generated.")
        return

    # --- Rank and recommend ---
    ranked = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'─'*50}")
    print("  SIGNAL RANKING")
    print(f"{'─'*50}")
    for i, (t, r) in enumerate(ranked, 1):
        tag = "  ← BUY" if i <= TOP_K and r > 0 else ""
        print(f"  {i}. {t:>6}  {r:+.2%}{tag}")

    # --- Update virtual portfolio ---
    portfolio = _load_portfolio()
    current_prices = {t: df["close"].iloc[-1] for t, df in ticker_data.items()}

    # Mark existing positions to market
    portfolio_value = portfolio["cash"]
    for t, pos in portfolio["positions"].items():
        price = current_prices.get(t, pos["entry_price"])
        portfolio_value += pos["shares"] * price

    # Liquidate all positions
    for t, pos in list(portfolio["positions"].items()):
        price = current_prices.get(t, pos["entry_price"])
        proceeds = pos["shares"] * price
        pnl = proceeds - pos["shares"] * pos["entry_price"]
        portfolio["cash"] += proceeds
        portfolio["trade_log"].append({
            "date": today, "ticker": t, "action": "sell",
            "shares": pos["shares"], "price": price, "pnl": round(pnl, 2),
        })
    portfolio["positions"] = {}

    # Buy top-K with positive signal
    top_picks = [t for t, r in ranked[:TOP_K] if r > 0]
    if top_picks:
        alloc = portfolio["cash"] / len(top_picks)
        for t in top_picks:
            price = current_prices[t]
            shares = alloc / price
            portfolio["positions"][t] = {"shares": shares, "entry_price": price}
            portfolio["cash"] -= alloc
            portfolio["trade_log"].append({
                "date": today, "ticker": t, "action": "buy",
                "shares": round(shares, 4), "price": price, "pnl": 0,
            })

    # Snapshot portfolio value
    portfolio_value = portfolio["cash"] + sum(
        pos["shares"] * current_prices.get(t, pos["entry_price"])
        for t, pos in portfolio["positions"].items()
    )
    portfolio["history"].append({"date": today, "value": round(portfolio_value, 2)})

    _save_portfolio(portfolio)

    print(f"\n  Virtual Portfolio Value: ${portfolio_value:,.2f}")
    print(f"  Cash: ${portfolio['cash']:,.2f}")
    if portfolio["positions"]:
        print("  Open Positions:")
        for t, pos in portfolio["positions"].items():
            val = pos["shares"] * current_prices.get(t, pos["entry_price"])
            print(f"    {t}: {pos['shares']:.2f} shares @ ${pos['entry_price']:.2f}  (value ${val:,.2f})")
    print(f"\n  Portfolio saved → {PORTFOLIO_FILE}")


if __name__ == "__main__":
    run_paper_trade()
