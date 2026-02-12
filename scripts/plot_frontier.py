#!/usr/bin/env python3
"""Plot efficient frontier from CSV output.

Usage:
    python scripts/plot_frontier.py results/optimize/frontier.csv
    python scripts/plot_frontier.py results/optimize/frontier.csv --output frontier.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot efficient frontier")
    parser.add_argument("csv_path", help="Path to frontier.csv")
    parser.add_argument("--output", "-o", default=None,
                        help="Output image path (default: show interactively)")
    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.csv_path, delimiter=",", names=True)
    except Exception as e:
        print(f"Error reading {args.csv_path}: {e}")
        sys.exit(1)

    returns = data["achieved_return"] * 100  # Percent.
    cvars = data["cvar"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cvars, returns, "b-o", markersize=7, linewidth=2, zorder=3)
    ax.set_xlabel("CVaR (95%) [%]", fontsize=12)
    ax.set_ylabel("Expected Return [%]", fontsize=12)
    ax.set_title("Mean-CVaR Efficient Frontier", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Annotate endpoints.
    ax.annotate(f"Min risk\n{cvars[0]:.2f}%",
                xy=(cvars[0], returns[0]),
                textcoords="offset points", xytext=(15, -10), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
    ax.annotate(f"Max return\n{returns[-1]:.2f}%",
                xy=(cvars[-1], returns[-1]),
                textcoords="offset points", xytext=(15, -10), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
