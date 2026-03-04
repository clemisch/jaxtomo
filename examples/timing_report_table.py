#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


def format_ms(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "-"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render timing_matrix JSON output as a Rich table."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="timing_matrix_results.json",
        help="Path to timing_matrix JSON output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))

    versions_data = data.get("versions", {})
    selected_versions = data.get("selected_versions")
    if isinstance(selected_versions, list) and selected_versions:
        versions = [v for v in selected_versions if v in versions_data]
    else:
        versions = list(versions_data.keys())

    table = Table(title=f"Timing Report: {input_path}")
    table.add_column("version")
    table.add_column("FP (ms)")
    table.add_column("BP (ms)")

    for version in versions:
        entry = versions_data.get(version, {})
        result = entry.get("result", {}) if isinstance(entry, dict) else {}
        fp = result.get("fp", {}) if isinstance(result, dict) else {}
        bp = result.get("bp", {}) if isinstance(result, dict) else {}
        fp_ms = format_ms(fp.get("milliseconds") if isinstance(fp, dict) else None)
        bp_ms = format_ms(bp.get("milliseconds") if isinstance(bp, dict) else None)
        table.add_row(str(version), fp_ms, bp_ms)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    main()
