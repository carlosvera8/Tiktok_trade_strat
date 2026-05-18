"""
run_backtest.py — Historical validation of the Kronos trading strategy.

Usage:
    python run_backtest.py

What it does:
    1. Downloads historical OHLCV for all tickers in config.TICKERS
    2. Runs a rolling-window backtest using Kronos predictions as signals
    3. Prints performance metrics (Sharpe, CAGR, drawdown, win rate)
    4. Saves equity curve and signal heatmap charts to reports/output/
"""
import sys
sys.path.insert(0, "kronos")

from config import (
    TICKERS, BENCHMARK_TICKER, BACKTEST_START, BACKTEST_END,
    MODEL_NAME, TOKENIZER_NAME, DEVICE,
)
from data.fetcher import fetch_all
from backtest.engine import run_backtest
from backtest.metrics import print_summary
from reports.visualizer import plot_equity_curve, plot_signal_heatmap


def main():
    print("=" * 50)
    print("  Kronos Backtest - Historical Validation")
    print("=" * 50)
    print(f"\n  Tickers   : {', '.join(TICKERS)}")
    print(f"  Period    : {BACKTEST_START} to {BACKTEST_END}")
    print(f"  Model     : {MODEL_NAME}\n")

    # 1. Fetch data
    print("Downloading market data ...")
    all_tickers = list(set(TICKERS + [BENCHMARK_TICKER]))
    ticker_data = fetch_all(all_tickers, start=BACKTEST_START, end=BACKTEST_END)

    if len(ticker_data) < 2:
        print("Not enough data to run backtest. Check ticker list and dates.")
        return

    # 2. Run backtest
    print("\nRunning Kronos backtest (this may take several minutes) ...")
    result = run_backtest(
        ticker_data=ticker_data,
        model_name=MODEL_NAME,
        tokenizer_name=TOKENIZER_NAME,
        device=DEVICE,
        benchmark_ticker=BENCHMARK_TICKER,
    )

    # 3. Print metrics
    print_summary(result.equity_curve, result.trade_log, label="Kronos Strategy")

    if not result.benchmark_curve.empty:
        print_summary(result.benchmark_curve, [], label=f"{BENCHMARK_TICKER} Buy & Hold")

    # 4. Save charts
    print("Saving charts ...")
    plot_equity_curve(result.equity_curve, result.benchmark_curve)
    if result.prediction_log:
        plot_signal_heatmap(result.prediction_log)

    print("\nDone. See reports/output/ for charts.")


if __name__ == "__main__":
    main()
