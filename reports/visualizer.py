import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

from config import REPORTS_DIR

sns.set_theme(style="darkgrid")


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series | None = None,
    title: str = "Portfolio vs Benchmark",
    save: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))

    # Normalize both curves to 100 at start
    norm_eq = equity_curve / equity_curve.iloc[0] * 100
    ax.plot(norm_eq.index, norm_eq.values, label="Kronos Strategy", color="#2196F3", linewidth=2)

    if benchmark_curve is not None and not benchmark_curve.empty:
        bm_aligned = benchmark_curve.reindex(equity_curve.index, method="ffill").dropna()
        if not bm_aligned.empty:
            norm_bm = bm_aligned / bm_aligned.iloc[0] * 100
            ax.plot(norm_bm.index, norm_bm.values, label="Buy & Hold", color="#FF9800", linewidth=2, linestyle="--")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Value (100 = start)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend()
    plt.tight_layout()

    out = REPORTS_DIR / "equity_curve.png"
    if save:
        fig.savefig(out, dpi=150)
        print(f"  Chart saved → {out}")
    plt.close(fig)
    return out


def plot_predictions(
    actual_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    ticker: str,
    save: bool = True,
) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    context_close = actual_df["close"].iloc[-60:]  # last 60 candles of context
    ax1.plot(context_close.index, context_close.values, label="Actual (context)", color="steelblue", linewidth=1.5)

    if not predicted_df.empty:
        predicted_df.index = pd.date_range(
            start=context_close.index[-1] + pd.Timedelta(days=1),
            periods=len(predicted_df), freq="B"
        )
        ax1.plot(predicted_df.index, predicted_df["close"].values, label="Kronos Forecast", color="tomato", linewidth=1.5, linestyle="--")

    ax1.set_ylabel("Close Price")
    ax1.set_title(f"{ticker} — Kronos Price Forecast", fontsize=13, fontweight="bold")
    ax1.legend()

    context_vol = actual_df["volume"].iloc[-60:]
    ax2.bar(context_vol.index, context_vol.values, label="Volume (actual)", color="steelblue", alpha=0.6, width=1)
    if not predicted_df.empty and "volume" in predicted_df.columns:
        ax2.bar(predicted_df.index, predicted_df["volume"].values, label="Volume (forecast)", color="tomato", alpha=0.6, width=1)
    ax2.set_ylabel("Volume")
    ax2.legend()

    plt.xticks(rotation=30)
    plt.tight_layout()

    out = REPORTS_DIR / f"prediction_{ticker}.png"
    if save:
        fig.savefig(out, dpi=150)
        print(f"  Chart saved → {out}")
    plt.close(fig)
    return out


def plot_signal_heatmap(prediction_log: list[dict], save: bool = True) -> Path:
    """Heatmap of predicted returns over time for each ticker."""
    df = pd.DataFrame(prediction_log)
    if df.empty:
        return REPORTS_DIR / "heatmap.png"

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="ticker", values="predicted_return", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot.columns))))
    sns.heatmap(
        pivot.T, center=0, cmap="RdYlGn", ax=ax,
        linewidths=0.3, cbar_kws={"label": "Predicted Return"},
        fmt=".1%",
    )
    ax.set_title("Kronos Signal Heatmap (Predicted Returns)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    plt.tight_layout()

    out = REPORTS_DIR / "signal_heatmap.png"
    if save:
        fig.savefig(out, dpi=150)
        print(f"  Chart saved → {out}")
    plt.close(fig)
    return out
