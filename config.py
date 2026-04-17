from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent
DATA_CACHE_DIR = ROOT / "data" / "cache"
REPORTS_DIR = ROOT / "reports" / "output"
PORTFOLIO_FILE = ROOT / "paper_trade" / "portfolio.json"

DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)

# --- Model ---
MODEL_NAME = "NeoQuasar/Kronos-small"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
DEVICE = "cpu"  # change to "cuda" if GPU is available

# --- Strategy ---
TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "SPY"]
BENCHMARK_TICKER = "SPY"

# Backtest window
BACKTEST_START = "2022-01-01"
BACKTEST_END = "2024-12-31"

# Kronos context/prediction settings
LOOKBACK = 400      # candles of history fed to Kronos
PRED_LEN = 5        # candles to predict ahead (~1 trading week)
REBALANCE_EVERY = 5 # rebalance portfolio every N candles

# Portfolio
INITIAL_CAPITAL = 10_000.0
TOP_K = 2           # hold top-K ranked tickers at any time

# Kronos sampling params (controls randomness)
TEMPERATURE = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 3    # average N samples for more stable predictions
