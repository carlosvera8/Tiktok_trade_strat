"""
daily_email.py — Run the paper trade and email the results.

Setup (one-time):
  1. Enable 2-Step Verification on your Google account.
  2. Go to https://myaccount.google.com/apppasswords and create an App Password.
  3. Set two environment variables (or paste directly into email_creds.py):
       GMAIL_USER     your Gmail address
       GMAIL_PASS     the 16-character App Password (no spaces)

Run manually:
  python daily_email.py
"""
import sys
sys.path.insert(0, "kronos")

import io
import json
import os
import smtplib
import traceback
from contextlib import redirect_stdout
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from config import PORTFOLIO_FILE, TICKERS

# ---------------------------------------------------------------------------
# Credentials — loaded from env vars or a local email_creds.py
# ---------------------------------------------------------------------------
try:
    import email_creds  # type: ignore
    GMAIL_USER = email_creds.GMAIL_USER
    GMAIL_PASS = email_creds.GMAIL_PASS
except ImportError:
    GMAIL_USER = os.environ.get("GMAIL_USER", "")
    GMAIL_PASS = os.environ.get("GMAIL_PASS", "")


def _run_paper_trade_captured() -> str:
    """Run the paper trade simulator and capture all stdout output."""
    from paper_trade.simulator import run_paper_trade
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            run_paper_trade()
    except Exception:
        buf.write("\n\n[ERROR during paper trade]\n")
        buf.write(traceback.format_exc())
    return buf.getvalue()


def _portfolio_summary() -> dict:
    """Return key portfolio stats from portfolio.json."""
    if not PORTFOLIO_FILE.exists():
        return {}
    with open(PORTFOLIO_FILE) as f:
        p = json.load(f)
    history = p.get("history", [])
    current_value = history[-1]["value"] if history else None
    initial = history[0]["value"] if history else None
    total_return = None
    if current_value and initial:
        total_return = (current_value - initial) / initial
    return {
        "value": current_value,
        "total_return": total_return,
        "cash": p.get("cash"),
        "positions": p.get("positions", {}),
        "days_tracked": len(history),
    }


def _build_email_body(trade_output: str, stats: dict) -> str:
    today = date.today().strftime("%B %d, %Y")
    lines = [f"Kronos Paper Trade — {today}", "=" * 50, ""]

    if stats.get("value") is not None:
        lines.append(f"Portfolio Value : ${stats['value']:,.2f}")
        lines.append(f"Cash            : ${stats['cash']:,.2f}")
        if stats.get("total_return") is not None:
            sign = "+" if stats["total_return"] >= 0 else ""
            lines.append(f"Total Return    : {sign}{stats['total_return']:.2%}  (since day 1)")
        lines.append(f"Days Tracked    : {stats['days_tracked']}")
        if stats["positions"]:
            lines.append("\nOpen Positions:")
            for ticker, pos in stats["positions"].items():
                lines.append(f"  {ticker}: {pos['shares']:.4f} shares @ ${pos['entry_price']:.2f}")
        lines.append("")

    lines.append("--- Full Run Output ---")
    lines.append(trade_output)
    return "\n".join(lines)


def send_email(subject: str, body: str) -> None:
    if not GMAIL_USER or not GMAIL_PASS:
        raise RuntimeError(
            "Missing credentials. Set GMAIL_USER and GMAIL_PASS env vars, "
            "or create email_creds.py (see daily_email.py header)."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_USER
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, GMAIL_USER, msg.as_string())


if __name__ == "__main__":
    print("Running paper trade ...")
    output = _run_paper_trade_captured()
    stats = _portfolio_summary()

    today_str = date.today().strftime("%Y-%m-%d")
    subject = f"[Kronos] Daily Signals — {today_str}"
    body = _build_email_body(output, stats)

    print("Sending email ...")
    try:
        send_email(subject, body)
        print(f"Email sent to {GMAIL_USER}")
    except Exception as e:
        print(f"Email failed: {e}")
        print("\n--- Trade output (not emailed) ---")
        print(output)
