#!/usr/bin/env python3
"""Plot backtest equity curves from CSV output.

Usage:
    python scripts/plot_backtest.py results/backtest/
    python scripts/plot_backtest.py results/backtest/ --output equity_curves.png
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def load_equity_curve(csv_path):
    """Load equity curve CSV, return dates and portfolio values."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")
    dates = data["date"]
    values = data["portfolio_value"]
    return dates, values


def parse_dates(date_strings):
    """Try to parse date strings into datetime objects."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return [datetime.strptime(d, fmt) for d in date_strings]
        except ValueError:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot backtest equity curves")
    parser.add_argument("result_dir", help="Directory containing *_equity.csv files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output image path (default: show interactively)")
    args = parser.parse_args()

    pattern = os.path.join(args.result_dir, "*_equity.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No *_equity.csv files found in {args.result_dir}")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for csv_path in csv_files:
        # Extract strategy name from filename.
        basename = os.path.basename(csv_path)
        strategy = basename.replace("_equity.csv", "")

        dates, values = load_equity_curve(csv_path)

        # Try to parse real dates for x-axis.
        parsed = parse_dates(dates)
        if parsed is not None:
            ax.plot(parsed, values, linewidth=1.5, label=strategy)
        else:
            x = np.arange(len(dates))
            ax.plot(x, values, linewidth=1.5, label=strategy)

    # Format x-axis based on whether we have real dates.
    if parsed is not None:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate(rotation=45)
    else:
        ax.set_xlabel("Trading Day", fontsize=12)

    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_title("Strategy Comparison: Equity Curves", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Format y-axis as currency.
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
