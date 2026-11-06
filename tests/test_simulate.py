"""Automatic tests for the simulator. Run from the project folder with:  pytest

Every function whose name starts with test_ is found and run by pytest.
A test passes if it finishes without error; an `assert` that is False makes it fail.
"""

import numpy as np   # numpy: arrays and comparisons
import pytest        # pytest: the test runner; pytest.raises checks that an error happens

from src.config import CONFIG, validate_config            # settings and their check (step 1)
from src.simulate import run_portfolio, simulate_returns  # the functions we test (steps 4-6)

# Small, made-up inputs so the tests are fast and don't need internet or downloaded data.
MU = np.array([0.002, 0.0005])            # two assets: a "stock" and a "bond"
COV = np.array([[0.0004, 0.00002],        # variances on the diagonal, covariance off it
                [0.00002, 0.0001]])
WEIGHTS = [0.6, 0.4]                      # 60/40 portfolio


def test_zero_returns_give_start_plus_contributions():
    """Zero returns and zero volatility: final value = start + weeks x contribution, exactly."""
    n_sims, n_weeks = 3, 52                                  # 3 simulations of 1 year
    zero_returns = np.zeros((n_sims, n_weeks, 2))            # every return is exactly 0 (so no volatility)
    values, contributed = run_portfolio(zero_returns, WEIGHTS, start_value=1000, weekly_contribution=25)

    expected = 1000 + 52 * 25                                # = 2300
    assert np.allclose(values[:, -1], expected)              # every simulation ends at 2300
    assert contributed == expected                           # and that is all your own money


def test_zero_volatility_cannot_use_cholesky():
    """A covariance matrix of zeros is not positive definite, so we expect a clear ValueError."""
    with pytest.raises(ValueError, match="positive definite"):   # the test PASSES only if this error happens
        simulate_returns(MU, np.zeros((2, 2)), n_weeks=10, n_sims=5, seed=1)


def test_constant_one_percent_grows_by_1_01_each_week():
    """No contributions and +1% every week: each week's value is exactly 1.01 x the previous one."""
    n_sims, n_weeks = 2, 20
    log_r = np.log(1.01)                                     # a +1% simple return as a log return
    returns = np.full((n_sims, n_weeks, 2), log_r)           # np.full: array filled with one value
    values, _ = run_portfolio(returns, WEIGHTS, start_value=1000, weekly_contribution=0)  # _ = ignored

    growth = values[:, 1:] / values[:, :-1]                  # each week divided by the week before
    assert np.allclose(growth, 1.01)                         # every ratio is 1.01
    assert np.isclose(values[0, -1], 1000 * 1.01 ** n_weeks) # and the end value is 1000 x 1.01^20


def test_same_seed_gives_identical_results():
    """Same seed -> exactly the same numbers; a different seed -> different numbers."""
    a = simulate_returns(MU, COV, n_weeks=30, n_sims=50, seed=42)
    b = simulate_returns(MU, COV, n_weeks=30, n_sims=50, seed=42)
    c = simulate_returns(MU, COV, n_weeks=30, n_sims=50, seed=7)

    assert np.array_equal(a, b)                              # identical, bit for bit
    assert not np.array_equal(a, c)                          # a different seed must change something


def test_output_shapes():
    """simulate_returns gives (n_sims, n_weeks, n_assets); run_portfolio gives (n_sims, n_weeks + 1)."""
    n_sims, n_weeks = 40, 26
    sim = simulate_returns(MU, COV, n_weeks=n_weeks, n_sims=n_sims, seed=0)
    values, _ = run_portfolio(sim, WEIGHTS, start_value=500, weekly_contribution=10)

    assert sim.shape == (n_sims, n_weeks, 2)
    assert values.shape == (n_sims, n_weeks + 1)
    assert np.all(values[:, 0] == 500)                       # column 0 is the start value


def test_weights_not_summing_to_one_raise_clear_error():
    """validate_config must refuse weights that don't add up to 1, with a helpful message."""
    bad = dict(CONFIG)                                       # a copy, so we don't change the real CONFIG
    bad["weights"] = [0.05] * len(CONFIG["tickers"])         # 19 x 0.05 = 0.95, not 1

    with pytest.raises(ValueError, match="must add up to 1"):
        validate_config(bad)
