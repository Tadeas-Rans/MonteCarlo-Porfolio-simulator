"""Tests for the analysis functions (today's money, own money, savings comparison).

Run from the project folder with:  pytest
"""

import numpy as np   # numpy: arrays and comparisons

from src.analyze import compare_to_savings, own_money, to_todays_money


def test_value_growing_with_inflation_is_constant_in_todays_money():
    """Something that grows exactly with inflation keeps the same buying power: 100 every week."""
    years = np.arange(105) / 52                       # week 0 to week 104, in years
    nominal = 100 * 1.02 ** years                     # grows by exactly 2% per year
    assert np.allclose(to_todays_money(nominal, 0.02), 100)


def test_to_todays_money_works_on_many_simulations():
    """A 2D array (n_sims, n_weeks + 1): every row is converted the same way."""
    values = np.full((3, 53), 102.0)                  # 3 simulations, 1 year, always 102
    real = to_todays_money(values, 0.02)
    assert np.isclose(real[0, 0], 102)                # week 0: today, nothing to convert
    assert np.allclose(real[:, -1], 100)              # after 1 year: 102 / 1.02 = 100


def test_own_money_without_inflation_is_running_total():
    """Start 1000, then 10, 20, 30: the path is 1000, 1010, 1030, 1060."""
    path = own_money(1000, [10, 20, 30])
    assert np.allclose(path, [1000, 1010, 1030, 1060])


def test_compare_to_savings_known_answer():
    """Four portfolio outcomes 90, 110, 120, 130 against savings ending at 100."""
    values = np.array([[0, 90], [0, 110], [0, 120], [0, 130]], dtype=float)  # only the last column matters
    savings = np.array([0.0, 100.0])
    result = compare_to_savings(values, savings)

    assert result["savings_final"] == 100
    assert result["prob_beat"] == 0.75                # 3 of the 4 end above 100
    assert np.isclose(result["extra_p50"], 15)        # median of -10, 10, 20, 30 is 15
