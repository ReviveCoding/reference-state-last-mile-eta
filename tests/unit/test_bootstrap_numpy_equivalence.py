import numpy as np
import pandas as pd
import pytest

from reference_eta.evaluation.bootstrap import clustered_mae_difference_ci


@pytest.mark.parametrize("seed", [0, 17, 902])
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("duplicate_index", [False, True])
@pytest.mark.parametrize("reserved_name", [False, True])
def test_clustered_mae_preserves_dataframe_resampling(
    seed, categorical, duplicate_index, reserved_name, monkeypatch
):
    rng = np.random.default_rng(51)
    count = 103
    frame = pd.DataFrame(
        {
            "courier_id": rng.choice(["c", "a", "b"], count),
            "work_date": rng.choice(["2026-01-03", "2026-01-01"], count),
        }
    )
    if categorical:
        frame["courier_id"] = pd.Categorical(
            frame["courier_id"], categories=["a", "b", "c", "unused"]
        )
    if duplicate_index:
        frame.index = np.arange(count) % 7
    truth, baseline, challenger = rng.normal(size=(3, count))
    if reserved_name:
        frame = frame.rename(columns={"courier_id": "baseline"})
    cluster_columns = ("baseline" if reserved_name else "courier_id", "work_date")
    working = frame.copy()
    working["y_true"], working["baseline"], working["challenger"] = truth, baseline, challenger
    clusters = list(working.groupby(list(cluster_columns), sort=False, dropna=False))
    reference_rng = np.random.default_rng(seed)
    draws = []
    for _ in range(37):
        selected = reference_rng.integers(0, len(clusters), size=len(clusters))
        sampled = pd.concat([clusters[index][1] for index in selected], ignore_index=True)
        draws.append(
            float(
                np.mean(np.abs(sampled["y_true"] - sampled["challenger"]))
                - np.mean(np.abs(sampled["y_true"] - sampled["baseline"]))
            )
        )
    expected = {
        "challenger_minus_baseline_mae": float(np.mean(draws)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
    }
    quantile = np.quantile

    def verify_draws(values, probability):
        np.testing.assert_array_equal(values, draws)
        return quantile(values, probability)

    monkeypatch.setattr(np, "quantile", verify_draws)
    assert (
        clustered_mae_difference_ci(
            frame,
            truth,
            baseline,
            challenger,
            cluster_columns=cluster_columns,
            n_bootstrap=37,
            seed=seed,
        )
        == expected
    )
