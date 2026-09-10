"""Validate the compact publication result bundle without rerunning simulations."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a result table and require a header."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader)


def verify() -> dict[str, int]:
    """Check campaign cardinalities, reference modes and path sanitization."""
    counts = {
        "baseline": len(csv_rows(RESULTS / "baseline_75" / "metrics.csv")),
        "screening": len(
            csv_rows(RESULTS / "applicability_99" / "screening" / "metrics.csv")
        ),
        "bandwidth": len(
            csv_rows(RESULTS / "applicability_99" / "bandwidth" / "metrics.csv")
        ),
        "confirmation": len(
            csv_rows(
                RESULTS / "applicability_99" / "confirmation" / "metrics.csv"
            )
        ),
    }
    expected = {"baseline": 75, "screening": 36, "bandwidth": 18, "confirmation": 45}
    if counts != expected:
        raise ValueError(f"unexpected campaign sizes: {counts}")

    reference = json.loads(
        (RESULTS / "reference" / "mixed_holdout_103.json").read_text()
    )
    if set(reference["modes"]) != {"ignore", "estimate", "oracle"}:
        raise ValueError("reference case must contain MPC, EnvMPC and oracle modes")
    text = "\n".join(
        path.read_text(errors="ignore")
        for path in RESULTS.rglob("*")
        if path.is_file() and path.suffix in {".csv", ".json", ".xml"}
    )
    if "/Users/" in text or "snapshot.pkl" in text:
        raise ValueError("public results contain a local path or omitted snapshot reference")

    checksum_path = RESULTS / "CHECKSUMS.sha256"
    checksums = {}
    for line in checksum_path.read_text().splitlines():
        expected_hash, relative_path = line.split("  ", maxsplit=1)
        checksums[relative_path] = expected_hash
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in RESULTS.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(checksums) != actual_paths:
        raise ValueError("result checksum inventory does not match the result files")
    for relative_path, expected_hash in checksums.items():
        actual_hash = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"checksum mismatch: {relative_path}")
    return counts


def main() -> None:
    """Run validation and print one compact success line."""
    counts = verify()
    print(
        "result bundle OK: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )


if __name__ == "__main__":
    main()
