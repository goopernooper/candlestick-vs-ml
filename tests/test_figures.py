"""The paper figures render from results files (figures.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from figures import COST_MODELS, ENCODINGS, MODELS, cost_figure, grid_figure  # noqa: E402


def test_figures_render(tmp_path):
    rng = np.random.default_rng(0)
    names = [f"{m}_{e}" for m, _ in MODELS for e, _ in ENCODINGS] + ["pattern_miner", "knn_b3", "knn_b5", "reversal"]
    summary = pd.DataFrame({"mcc": rng.uniform(0, 0.035, len(names))}, index=names)
    grid_figure(summary, tmp_path / "grid.png", "test")
    rows = [{"model": m, "strategy": s, "cost_bps": c, "ann_return": 0.2 - 0.01 * c * (i + 1)}
            for i, m in enumerate([*COST_MODELS, "always_long"]) for s in ["long_short", "long_flat"]
            for c in [0, 1, 2, 5, 10, 20]]
    cost_figure(pd.DataFrame(rows), tmp_path / "cost.png", "test", headline=5)
    for name in ["grid.png", "cost.png"]:
        assert (tmp_path / name).stat().st_size > 20_000
