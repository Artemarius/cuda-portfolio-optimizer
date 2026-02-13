#!/usr/bin/env python3
"""Download real S&P 500 stock data for portfolio optimization demos.

Downloads daily adjusted close prices for S&P 500 stocks using yfinance,
and writes a wide-format CSV compatible with csv_loader.cpp.

Usage:
    pip install -r scripts/requirements.txt
    python scripts/download_data.py                  # 10 stocks (default)
    python scripts/download_data.py --universe 50    # 50 stocks
"""

import argparse
import sys
from pathlib import Path

import yfinance as yf

# 10 S&P 500 stocks across sectors (original set).
TICKERS_10 = ["AAPL", "MSFT", "NVDA", "GOOG", "JPM", "JNJ", "XOM", "PG", "AMZN", "V"]

# 50 S&P 500 stocks across all 11 GICS sectors.
TICKERS_50 = [
    # Technology (7)
    "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "CSCO",
    # Health Care (5)
    "UNH", "JNJ", "LLY", "PFE", "ABT",
    # Financials (6)
    "JPM", "BAC", "GS", "BRK-B", "V", "MA",
    # Consumer Discretionary (5)
    "AMZN", "TSLA", "HD", "MCD", "NKE",
    # Communication Services (3)
    "GOOG", "META", "DIS",
    # Industrials (5)
    "CAT", "HON", "UPS", "BA", "RTX",
    # Consumer Staples (4)
    "PG", "KO", "PEP", "COST",
    # Energy (4)
    "XOM", "CVX", "COP", "SLB",
    # Utilities (3)
    "NEE", "DUK", "SO",
    # Real Estate (3)
    "PLD", "AMT", "SPG",
    # Materials (3)
    "LIN", "APD", "SHW",
]

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"


def main():
    parser = argparse.ArgumentParser(
        description="Download S&P 500 stock data for portfolio optimization.")
    parser.add_argument(
        "--universe", type=int, choices=[10, 50], default=10,
        help="Number of stocks to download (default: 10)")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"

    if args.universe == 50:
        tickers = TICKERS_50
        output_path = data_dir / "sp500_50_prices.csv"
    else:
        tickers = TICKERS_10
        output_path = data_dir / "sp500_prices.csv"

    print(f"Downloading {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    df = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True)

    # yfinance returns MultiIndex columns (Price, Ticker) -- extract Close.
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df = df["Close"]

    # Ensure column order matches our ticker list.
    # Filter to only tickers that were actually downloaded.
    available = [t for t in tickers if t in df.columns]
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        print(f"WARNING: Missing tickers (not in download): {missing}")
    df = df[available]

    # Drop rows with any NaN (holidays, missing data).
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} rows with missing data")

    # Write wide-format CSV: Date,AAPL,MSFT,...
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.index.name = "Date"
    df.to_csv(output_path, float_format="%.4f")

    print(f"\nSummary:")
    print(f"  Rows:    {len(df)}")
    print(f"  Tickers: {list(df.columns)} ({len(df.columns)} stocks)")
    print(f"  From:    {df.index[0].strftime('%Y-%m-%d')}")
    print(f"  To:      {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Written: {output_path}")


if __name__ == "__main__":
    main()
