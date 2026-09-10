import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit


def test_public_result_bundle_is_complete_and_sanitized():
    subprocess.run([sys.executable, "scripts/verify_results.py"], check=True)
