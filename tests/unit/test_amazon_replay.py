from reference_eta.decisions.amazon_replay import demo_replay


def test_route_replay_generates_comparable_strategies() -> None:
    report = demo_replay(seed=3, n_stops=10)
    assert set(report["strategy"]) == {
        "recorded_proxy",
        "nearest_neighbor",
        "time_window_greedy",
        "time_window_two_opt",
    }
    assert (report["completion_minutes"] > 0).all()
    assert (report["time_window_violations"] >= 0).all()
    greedy = report.set_index("strategy").loc["time_window_greedy"]
    improved = report.set_index("strategy").loc["time_window_two_opt"]
    greedy_objective = (
        greedy["completion_minutes"]
        + 250.0 * greedy["time_window_violations"]
        + 5.0 * greedy["total_lateness_minutes"]
    )
    improved_objective = (
        improved["completion_minutes"]
        + 250.0 * improved["time_window_violations"]
        + 5.0 * improved["total_lateness_minutes"]
    )
    assert improved_objective <= greedy_objective + 1e-9
