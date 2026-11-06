"""Automatic tests for the simulator. Run from the project folder with:  pytest

Every function whose name starts with test_ is found and run by pytest.
A test passes if it finishes without error; an `assert` that is False makes it fail.
"""

import numpy as np   # numpy: arrays and comparisons
import pytest        # pytest: the test runner; pytest.raises checks that an error happens

from src.config import CONFIG, validate_config            # settings and their check (step 1)
from src.simulate import (                                # the functions we test (steps 4-6 + 12)
    contribution_schedule,
    run_portfolio,
    run_savings,
    simulate_returns,
)

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


# ---------------- Step 12: indexed contributions and the savings account ----------------

def test_contribution_schedule_raises_every_quarter():
    """With 2% inflation: weeks 1-13 pay the base, week 14 pays base x 1.02^(1/4), week 53 pays base x 1.02."""
    c = contribution_schedule(40, n_weeks=104, inflation=0.02, index_every_weeks=13)

    assert len(c) == 104                                     # one amount per week
    assert np.allclose(c[:13], 40)                           # first quarter: base amount
    assert np.isclose(c[13], 40 * 1.02 ** 0.25)              # second quarter: one quarter of inflation
    assert np.isclose(c[52], 40 * 1.02)                      # after exactly one year: one year of inflation


def test_contribution_schedule_without_inflation_is_flat():
    """Inflation 0 means no raises at all."""
    assert np.allclose(contribution_schedule(25, n_weeks=60, inflation=0.0), 25)


def test_array_contribution_same_as_number():
    """Passing 52 copies of 25 must give exactly the same result as passing the number 25."""
    sim = simulate_returns(MU, COV, n_weeks=52, n_sims=10, seed=3)
    a, total_a = run_portfolio(sim, WEIGHTS, 1000, 25)
    b, total_b = run_portfolio(sim, WEIGHTS, 1000, np.full(52, 25.0))

    assert np.allclose(a, b)
    assert np.isclose(total_a, total_b)


def test_savings_grow_by_exactly_the_rate_in_one_year():
    """No contributions, 3% interest, 52 weeks: the savings account ends at start x 1.03."""
    savings = run_savings(0.03, start_value=1000, weekly_contribution=0, n_weeks=52)

    assert savings.shape == (53,)                            # week 0 to week 52
    assert np.isclose(savings[-1], 1030)


def test_savings_at_zero_rate_is_just_your_money():
    """0% interest: the savings account holds exactly what you put in."""
    savings = run_savings(0.0, start_value=500, weekly_contribution=10, n_weeks=20)
    assert np.isclose(savings[-1], 500 + 20 * 10)
