"""Reproduce and validate the paper's complete mixed-wrench reference case."""

from __future__ import annotations

import json
from pathlib import Path

import click

from experiments.research.common import write_json
from experiments.research.paper_1_external_wrench.applicability import (
    confirmation_cases,
    run_stage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ID = "mixed_holdout_103"
DEFAULT_GOLDEN = PROJECT_ROOT / "results" / "reference" / f"{REFERENCE_ID}.json"
DEFAULT_SELECTION = (
    PROJECT_ROOT
    / "results"
    / "applicability_99"
    / "confirmation"
    / "selection.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "out" / "reference_case"
COMPARISON_METRICS = (
    "tracking_position_rmse_m",
    "tracking_orientation_rmse_rad",
    "control_effort_normalized2_s",
)


def validate_reference(
    actual_rows: list[dict], golden: dict, relative_tolerance: float
) -> dict:
    """Compare physical metrics and invariant outcomes with the archived case."""
    actual = {row["mode"]: row for row in actual_rows}
    expected = {
        mode: payload["summary"] for mode, payload in golden["modes"].items()
    }
    if set(actual) != {"ignore", "estimate", "oracle"}:
        raise ValueError(f"reference run returned unexpected modes: {sorted(actual)}")

    comparisons = []
    for mode in ("ignore", "estimate", "oracle"):
        if float(actual[mode]["solver_nonstandard_status_fraction"]) != 0.0:
            raise ValueError(f"{mode} produced a nonstandard acados status")
        for metric in COMPARISON_METRICS:
            observed = float(actual[mode][metric])
            reference = float(expected[mode][metric])
            relative_error = abs(observed - reference) / max(abs(reference), 1e-12)
            comparisons.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "actual": observed,
                    "reference": reference,
                    "relative_error": relative_error,
                    "within_tolerance": relative_error <= relative_tolerance,
                }
            )
    failed = [row for row in comparisons if not row["within_tolerance"]]
    if failed:
        details = ", ".join(
            f"{row['mode']}:{row['metric']}={100.0 * row['relative_error']:.2f}%"
            for row in failed
        )
        raise ValueError(f"reference metrics exceeded tolerance: {details}")
    if not (
        actual["estimate"]["tracking_position_rmse_m"]
        < actual["ignore"]["tracking_position_rmse_m"]
        and actual["estimate"]["tracking_orientation_rmse_rad"]
        < actual["ignore"]["tracking_orientation_rmse_rad"]
    ):
        raise ValueError("EnvMPC did not improve both tracking metrics over MPC")
    return {
        "case_id": REFERENCE_ID,
        "relative_tolerance": relative_tolerance,
        "comparisons": comparisons,
        "solver_time_p95_ms": {
            mode: float(row["solver_time_p95_ms"]) for mode, row in actual.items()
        },
        "status": "passed",
    }


@click.command(context_settings={"show_default": True})
@click.option(
    "--output", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_OUTPUT
)
@click.option("--force", is_flag=True, help="Recompute existing mode outputs.")
@click.option("--relative-tolerance", type=float, default=0.10)
def main(output: Path, force: bool, relative_tolerance: float) -> None:
    """Run MPC, EnvMPC and oracle modes and compare them with the publication."""
    selection = json.loads(DEFAULT_SELECTION.read_text())
    cases = [
        case
        for case in confirmation_cases(selection)
        if case.candidate_id == REFERENCE_ID
    ]
    rows = run_stage(cases, output, force=force)
    try:
        validation = validate_reference(
            rows, json.loads(DEFAULT_GOLDEN.read_text()), relative_tolerance
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    validation_path = output / "validation.json"
    write_json(validation_path, validation)
    click.echo(f"Reference case passed; validation saved to {validation_path}")


if __name__ == "__main__":
    main()
