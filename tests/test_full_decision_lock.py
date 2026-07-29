from backtest import reproduce_baseline as rb


def test_full_live_decision_recipe_is_locked():
    ok, actual, expected = rb.decision_lock_match()
    assert ok, f"full-decision drift: actual={actual} expected={expected}"
