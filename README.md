# Tiktok_trade_strat

A paper trading and backtesting system that uses the [Kronos](https://huggingface.co/NeoQuasar/Kronos-small) time-series forecasting model to rank stocks by predicted return and manage a virtual portfolio.

## What it does

- **Backtest** — runs the Kronos strategy over historical data (2022–2024 by default) and prints Sharpe ratio, CAGR, drawdown, and win rate, then saves equity curve and signal heatmap charts to `reports/output/`.
- **Paper trade** — downloads today's data, predicts returns for the next ~5 trading days, and updates a virtual portfolio tracked in `paper_trade/portfolio.json`. Run it daily to simulate trading without real money.

---

## Setup

### 1. Activate the virtual environment

**Windows (cmd / PowerShell):**
```
venv\Scripts\activate
```

**Mac / Linux:**
```
source venv/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

> **Note:** `torch` is a large download (~2 GB). The first run also downloads the Kronos model weights from Hugging Face, so expect a slow first start.

---

## Running the backtest

```
python run_backtest.py
```

This will:
1. Download historical OHLCV data for all tickers in `config.py` (cached to `data/cache/` after first download)
2. Run Kronos predictions in a rolling window — **this can take several minutes**
3. Print performance metrics to the terminal
4. Save charts to `reports/output/`

---

## Running the paper trader

```
python run_paper_trade.py
```

Or override which tickers to watch:

```
python run_paper_trade.py AAPL TSLA NVDA
```

The virtual portfolio is saved to `paper_trade/portfolio.json`. To start fresh, delete that file.

---

## Configuration

All settings live in [config.py](config.py):

| Setting | Default | Description |
|---|---|---|
| `TICKERS` | AAPL, MSFT, TSLA, GOOGL, SPY | Stocks to trade |
| `BENCHMARK_TICKER` | SPY | Buy-and-hold benchmark |
| `BACKTEST_START` | 2022-01-01 | Backtest start date |
| `BACKTEST_END` | 2024-12-31 | Backtest end date |
| `INITIAL_CAPITAL` | $10,000 | Starting portfolio value |
| `TOP_K` | 2 | Number of tickers to hold at once |
| `LOOKBACK` | 400 | Candles of history fed to Kronos |
| `PRED_LEN` | 5 | Candles predicted ahead (~1 week) |
| `REBALANCE_EVERY` | 5 | Rebalance every N candles |
| `DEVICE` | cpu | Change to `cuda` if you have a GPU |

---

## Project structure

```
├── config.py               # All settings
├── run_backtest.py         # Entry point: historical backtest
├── run_paper_trade.py      # Entry point: paper trading
├── data/
│   ├── fetcher.py          # Downloads and caches OHLCV via yfinance
│   └── cache/              # Auto-created CSV cache
├── backtest/
│   ├── engine.py           # Rolling-window backtest logic
│   └── metrics.py          # Sharpe, CAGR, drawdown calculations
├── paper_trade/
│   ├── simulator.py        # Live paper trading loop
│   └── portfolio.json      # Virtual portfolio state (auto-created)
└── reports/
    ├── visualizer.py       # Equity curve and heatmap charts
    └── output/             # Auto-created chart output folder
```
