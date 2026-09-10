from pathlib import Path

import pytest


PRIMARY_MARKERS = ("unit", "integration", "system", "regression")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Require one primary test level and keep regressions in their own directory."""
    for item in items:
        primary = [name for name in PRIMARY_MARKERS if item.get_closest_marker(name)]
        if len(primary) != 1:
            raise pytest.UsageError(
                f"{item.nodeid} must have exactly one primary marker, got {primary}"
            )
        in_regressions = "regressions" in Path(str(item.path)).parts
        if (primary[0] == "regression") != in_regressions:
            raise pytest.UsageError(
                f"{item.nodeid}: regression marker and tests/regressions location disagree"
            )
