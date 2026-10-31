from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_match_data
from data.synthetic import ACTIVE_TEAMS, generate_sales

MAX_CSV_BYTES = 2 * 1024 * 1024


def export_sample_csv(destination: str | Path | None = None) -> Path:
    output_path = Path(destination) if destination else PROJECT_ROOT / "sample_sales_data.csv"
    matches, match_source = load_match_data()
    end = date.today()
    sales = generate_sales(
        matches=matches,
        start=end - timedelta(days=730),
        end=end,
        teams=ACTIVE_TEAMS,
        random_seed=7,
        team_level_scale=(0.9, 1.1),
    )
    sales = sales.loc[:, ["date", "team", "product", "units", "price"]]
    sales["date"] = sales["date"].dt.strftime("%Y-%m-%d")
    sales["units"] = sales["units"].astype(int).clip(lower=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(output_path, index=False)
    size = output_path.stat().st_size
    if size > MAX_CSV_BYTES:
        output_path.unlink()
        raise ValueError(f"Exported CSV exceeds the 2 MB upload limit: {size} bytes.")
    print(
        f"Wrote {len(sales)} rows to {output_path} "
        f"using {match_source} match data ({size:,} bytes)."
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export match-linked sample sales data.")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="CSV destination (defaults to sample_sales_data.csv in the project root).",
    )
    args = parser.parse_args()
    export_sample_csv(args.output)


if __name__ == "__main__":
    main()
