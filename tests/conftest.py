"""Shared test setup.

Tests always run with the holdout locked, whatever config.yaml says. The holdout
was unlocked once for the final run (DECISIONS.md D-031), and without this the
guard tests would start a real holdout run on the full data instead of checking
that it is refused.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import load_config  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def holdout_locked():
    # Session scope, so module-scoped fixtures that build tables also see it.
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(load_config()["splits"], "unlock_holdout", False)
        yield
