#!/usr/bin/env python3
"""Download real S&P 500 stock data for portfolio optimization demos.

Downloads daily adjusted close prices for 10 multi-sector S&P 500 stocks
using yfinance, and writes a wide-format CSV compatible with csv_loader.cpp.

Usage:
    pip install -r scripts/requirements.txt
    python scripts/download_data.py
"""

import sys
from pathlib import Path

import yfinance as yf

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sp500_prices.csv"

# 10 S&P 500 stocks across sectors.
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOG", "JPM", "JNJ", "XOM", "PG", "AMZN", "V"]

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"


def main():
    print(f"Downloading {len(TICKERS)} tickers: {', '.join(TICKERS)}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    df = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)

    # yfinance returns MultiIndex columns (Price, Ticker) -- extract Close.
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df = df["Close"]

    # Ensure column order matches our ticker list.
    df = df[TICKERS]

    # Drop rows with any NaN (holidays, missing data).
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} rows with missing data")

    # Write wide-format CSV: Date,AAPL,MSFT,...
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.index.name = "Date"
    df.to_csv(OUTPUT_PATH, float_format="%.4f")

    print(f"\nSummary:")
    print(f"  Rows:    {len(df)}")
    print(f"  Tickers: {list(df.columns)}")
    print(f"  From:    {df.index[0].strftime('%Y-%m-%d')}")
    print(f"  To:      {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
