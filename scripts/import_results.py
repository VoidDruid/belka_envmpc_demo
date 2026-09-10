"""Import the compact, public result subset from the complete article archive."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


REFERENCE_CASE = "mixed_holdout_103"


def clean_json(value):
    """Remove machine-local paths and references to intentionally omitted snapshots."""
    if isinstance(value, dict):
        cleaned = {
            key: clean_json(item)
            for key, item in value.items()
            if key not in {"output_dir", "runtime_snapshots"}
        }
        if isinstance(cleaned.get("artifacts"), list):
            cleaned["artifacts"] = [
                item for item in cleaned["artifacts"] if item != "snapshot.pkl"
            ]
        return cleaned
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def copy_json(source: Path, target: Path) -> None:
    """Copy one JSON document after applying the public-artifact filter."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(clean_json(json.loads(source.read_text())), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def copy_csv(source: Path, target: Path) -> None:
    """Copy one CSV table without its machine-local output directory column."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as source_handle:
        reader = csv.DictReader(source_handle)
        rows = list(reader)
        fieldnames = [name for name in (reader.fieldnames or []) if name != "output_dir"]
    if not rows:
        shutil.copy2(source, target)
        return
    with target.open("w", newline="", encoding="utf-8") as target_handle:
        writer = csv.DictWriter(target_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: row.get(name, "") for name in fieldnames} for row in rows
        )


def import_results(source_root: Path, target_root: Path) -> None:
    """Create the compact result tree documented by the public repository."""
    baseline = source_root / "baseline_75"
    for name in ("aggregate_metrics.csv", "metrics.csv"):
        copy_csv(baseline / name, target_root / "baseline_75" / name)
    copy_json(baseline / "summary.json", target_root / "baseline_75" / "summary.json")

    applicability = source_root / "applicability_99"
    for stage, names in {
        "screening": ("manifest.json", "metrics.csv", "selection.json"),
        "bandwidth": ("manifest.json", "metrics.csv", "selection.json"),
        "confirmation": (
            "effects.csv",
            "effects.json",
            "manifest.json",
            "metrics.csv",
            "selection.json",
        ),
    }.items():
        for name in names:
            source = applicability / stage / name
            target = target_root / "applicability_99" / stage / name
            (copy_csv if source.suffix == ".csv" else copy_json)(source, target)
    shutil.copy2(
        applicability / "envmpc_applicability.jpg",
        target_root / "applicability_99" / "envmpc_applicability.jpg",
    )
    copy_json(
        applicability / "confirmation" / "full_case_mixed_103.json",
        target_root / "reference" / f"{REFERENCE_CASE}.json",
    )
    for mode in ("ignore", "estimate", "oracle"):
        source_mode = applicability / "confirmation" / REFERENCE_CASE / mode
        target_mode = target_root / "reference" / REFERENCE_CASE / mode
        for name in ("manifest.json", "scenario.json", "summary.json"):
            copy_json(source_mode / name, target_mode / name)
        copy_csv(source_mode / "metrics.csv", target_mode / "metrics.csv")
        shutil.copy2(source_mode / "model.xml", target_mode / "model.xml")


def main() -> None:
    """Parse paths and import the selected publication artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="directory containing baseline_75 and applicability_99")
    parser.add_argument("--target", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    import_results(args.source.resolve(), args.target.resolve())


if __name__ == "__main__":
    main()
