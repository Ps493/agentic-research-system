"""Command-line entry point.

Usage:
    python -m src.cli --goal "Summarize the state of solid-state EV batteries in 2026"
    python -m src.cli --goal "..." --out reports/my_report.md
"""
from __future__ import annotations

import argparse
import os
import sys

from .agent import ResearchAgent
from . import config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic Research System")
    parser.add_argument("--goal", required=True, help="The research goal/task to give the agent.")
    parser.add_argument("--out", default=None, help="Path to write the final Markdown report.")
    args = parser.parse_args(argv)

    agent = ResearchAgent()
    report, memory = agent.run(args.goal)

    out_path = args.out or os.path.join(config.REPORTS_DIR, "report.md")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)
    print(f"\nFull report saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
