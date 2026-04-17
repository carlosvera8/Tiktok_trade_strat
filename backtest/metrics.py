import numpy as np
import pandas as pd


def daily_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().dropna()


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.05) -> float:
    r = daily_returns(equity_curve)
    excess = r - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def cagr(equity_curve: pd.Series) -> float:
    years = len(equity_curve) / 252
    if years == 0:
        return 0.0
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1)


def win_rate(trade_log: list[dict]) -> float:
    if not trade_log:
        return 0.0
    wins = sum(1 for t in trade_log if t.get("pnl", 0) > 0)
    return wins / len(trade_log)


def total_return(equity_curve: pd.Series) -> float:
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def print_summary(equity_curve: pd.Series, trade_log: list[dict], label: str = "Strategy") -> None:
    print(f"\n{'='*40}")
    print(f"  {label} Performance Summary")
    print(f"{'='*40}")
    print(f"  Total Return    : {total_return(equity_curve):>8.2%}")
    print(f"  CAGR            : {cagr(equity_curve):>8.2%}")
    print(f"  Sharpe Ratio    : {sharpe_ratio(equity_curve):>8.2f}")
    print(f"  Max Drawdown    : {max_drawdown(equity_curve):>8.2%}")
    print(f"  Win Rate        : {win_rate(trade_log):>8.2%}")
    print(f"  Total Trades    : {len(trade_log):>8}")
    print(f"{'='*40}\n")
