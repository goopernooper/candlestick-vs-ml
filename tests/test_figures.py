"""The paper figures render from results files (figures.py), and the paper assembles (paper/build.py)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figures import (COST_MODELS, ENCODING, ENCODINGS, EPISODE_MODELS, FAMILY, FOCUS, MODELS,  # noqa: E402
                     confident_figure, cost_figure, coverage_figure, episodes_figure, grid_figure, name)


def test_figures_render(tmp_path):
    rng = np.random.default_rng(0)
    names = [f"{m}_{e}" for m, _ in MODELS for e, _ in ENCODINGS] + ["pattern_miner", "knn_b3", "knn_b5", "reversal"]
    summary = pd.DataFrame({"mcc": rng.uniform(0, 0.035, len(names))}, index=names)
    grid_figure(summary, tmp_path / "grid.png", "test")
    rows = [{"model": m, "strategy": s, "cost_bps": c, "ann_return": 0.2 - 0.01 * c * (i + 1)}
            for i, m in enumerate([*COST_MODELS, "always_long"]) for s in ["long_short", "long_flat"]
            for c in [0, 1, 2, 5, 10, 20]]
    cost_figure(pd.DataFrame(rows), tmp_path / "cost.png", "test", headline=5)

    models = [*names, "lightgbm_today", "always_long"]
    cov = pd.DataFrame([{"model": m, "target_coverage": c, "coverage": c, "mcc": rng.uniform(0, 0.1)}
                        for m in models for c in [0.1, 0.25, 0.5, 1.0]])
    coverage_figure(cov, tmp_path / "coverage.png", "test")
    vs = cov.assign(diff=lambda d: d["mcc"] - 0.03, diff_low=lambda d: d["diff"] - 0.02,
                    diff_high=lambda d: d["diff"] + 0.02)
    confident_figure(vs, tmp_path / "confident.png", "test")
    episodes = ["2008-09", "2011", "2020", "2022", "other"]
    ep = pd.DataFrame([{"model": m, "episode": e, "mcc": rng.uniform(-0.03, 0.09), "episode_days": 100,
                        "share_of_high_days": 0.2} for m in EPISODE_MODELS for e in episodes])
    loo = pd.DataFrame([{"model": m, "dropped": d, "mcc": rng.uniform(0, 0.06)}
                        for m in EPISODE_MODELS for d in ["2008-09", "2011", "2020", "2022", "none"]])
    episodes_figure(ep, loo, tmp_path / "episodes.png", "test")
    for f in ["grid.png", "cost.png", "coverage.png", "confident.png", "episodes.png"]:
        assert (tmp_path / f).stat().st_size > 20_000


def test_readable_names():
    assert name("rf_ohlc") == "Random forest, normalized prices"
    assert name("svm_b") == "SVM, encoding B"
    assert name("lightgbm_today") == "LightGBM, today's return only"
    assert set(FOCUS) <= set(COST_MODELS)
    assert {m.split("_")[0] for m in ["lightgbm_a", "rf_a", "svm_a", "mlp_a"]} == set(FAMILY)
    assert set(ENCODING) == {e for e, _ in ENCODINGS}


def _build_module():
    spec = importlib.util.spec_from_file_location("build_paper", ROOT / "paper" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_merges_reference_lists(tmp_path):
    build = _build_module()
    (tmp_path / "01_a.md").write_text("## 1. A\n\nText (Smith, 2000).\n\n### References for this section\n\n"
                                      "- Smith, J. (2000). Title. *Journal*, 1, 1–2.\n")
    (tmp_path / "02_b.md").write_text("## 2. B\n\nMore (Abel, 1999; Smith, 2000).\n\n### References for these sections\n\n"
                                      "- Smith, J. (2000). Title. *Journal*, 1, 1–2.\n- Abel, K. (1999). Other. *J*, 2, 3.\n")
    text = build.build(tmp_path)
    assert "References for" not in text
    assert text.index("## 1. A") < text.index("## 2. B") < text.index("## References")
    refs = text.split("## References")[1]
    assert refs.count("Smith, J.") == 1 and refs.index("Abel") < refs.index("Smith")


def test_paper_has_no_placeholders():
    text = _build_module().build()
    for marker in ["[X]", "[YEAR]", "Figure X"]:
        assert marker not in text
    for k in range(1, 6):
        assert f"**Figure {k}." in text


def test_poster_figures_render_at_print_size(tmp_path):
    from poster_figures import DPI, poster_confident, poster_cost, poster_grid
    from PIL import Image
    rng = np.random.default_rng(1)
    names = [f"{m}_{e}" for m, _ in MODELS for e, _ in ENCODINGS] + ["reversal"]
    poster_grid(pd.DataFrame({"mcc": rng.uniform(0, 0.035, len(names))}, index=names), tmp_path / "g.png")
    rows = [{"model": m, "strategy": "long_short", "cost_bps": c, "ann_return": 0.2 - 0.01 * c * (i + 1)}
            for i, m in enumerate([*COST_MODELS, "always_long"]) for c in [0, 1, 2, 5, 10, 20]]
    poster_cost(pd.DataFrame(rows), tmp_path / "c.png", headline=5)
    vs = pd.DataFrame([{"model": m, "target_coverage": 0.1, "diff": d, "diff_low": d - 0.02, "diff_high": d + 0.02}
                       for m, d in zip(names, rng.uniform(-0.03, 0.06, len(names)))])
    poster_confident(vs, tmp_path / "f.png")
    # printed at 17.7 in and 12 in wide, at the print resolution
    assert Image.open(tmp_path / "g.png").size[0] == round(17.7 * DPI)
    assert Image.open(tmp_path / "c.png").size[0] == round(17.7 * DPI)
    assert Image.open(tmp_path / "f.png").size[0] == round(12.0 * DPI)
