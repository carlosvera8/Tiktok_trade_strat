"""
run_paper_trade.py — Paper trading simulation using live market data.

Usage:
    python run_paper_trade.py              # run with default tickers from config
    python run_paper_trade.py AAPL TSLA   # override tickers on the command line

What it does:
    1. Downloads today's latest OHLCV for each ticker
    2. Uses Kronos to predict returns over the next PRED_LEN trading days
    3. Ranks tickers by predicted return, recommends top-K buys
    4. Updates a virtual portfolio tracked in paper_trade/portfolio.json

Run this daily (or as often as you like) to simulate trading without real money.
To reset the portfolio, delete paper_trade/portfolio.json.
"""
import sys
sys.path.insert(0, "kronos")

from paper_trade.simulator import run_paper_trade

if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    run_paper_trade(tickers)
