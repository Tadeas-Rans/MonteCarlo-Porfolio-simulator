"""Tests for the history window (trim_history). No download needed: we build fake prices.

Run from the project folder with:  pytest
"""

import pandas as pd   # pandas: tables with dates

from src.data import trim_history


def fake_prices(years):
    """A table of weekly prices (all 1.0) covering `years` years, ending on 2026-10-26."""
    dates = pd.date_range(end="2026-10-26", periods=years * 52 + 1, freq="W-MON")   # one row per Monday
    return pd.DataFrame({"A": 1.0, "B": 1.0}, index=dates)


def test_none_keeps_everything():
    prices = fake_prices(8)
    assert len(trim_history(prices, None)) == len(prices)


def test_keeps_only_the_last_years():
    """12 years of data, ask for 5: the first kept week is at most 5 years before the last week."""
    trimmed = trim_history(fake_prices(12), 5)
    span = (trimmed.index[-1] - trimmed.index[0]).days / 365.25
    assert 4.98 < span <= 5.0
    assert trimmed.index[-1] == pd.Timestamp("2026-10-26")      # the newest data is always kept


def test_too_little_history_keeps_all_and_warns(capsys):
    """8 years of data, ask for 10: nothing is cut, and a warning is printed."""
    prices = fake_prices(8)
    trimmed = trim_history(prices, 10)
    assert len(trimmed) == len(prices)
    assert "Warning" in capsys.readouterr().out                  # capsys catches what was printed
