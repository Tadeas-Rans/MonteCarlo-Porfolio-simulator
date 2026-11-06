"""All settings for the portfolio Monte Carlo simulator live in this file."""

import math  # standard-library maths module; we need math.isclose() to compare decimals safely


# Each ETF ticker (as Yahoo Finance spells it) paired with its weight in the portfolio.
# A dictionary keeps every ticker right next to its own weight, so they can't get out of order.
PORTFOLIO = {
    "XDEW.DE": 0.05,  # Xtrackers S&P 500 Equal Weight
    "VEUR.AS": 0.05,  # Vanguard FTSE Developed Europe
    "IQQ0.DE": 0.05,  # iShares MSCI World Minimum Volatility
    "VHYL.AS": 0.05,  # Vanguard FTSE All-World High Dividend Yield
    "IS3S.DE": 0.05,  # iShares MSCI World Value Factor
    "IS3Q.DE": 0.05,  # iShares MSCI World Quality Factor
    "XDWH.DE": 0.05,  # Xtrackers MSCI World Health Care
    "XDWS.DE": 0.05,  # Xtrackers MSCI World Consumer Staples
    "IQQI.DE": 0.05,  # iShares Global Infrastructure
    "EUNH.DE": 0.10,  # iShares Core Euro Government Bond (the one 10% position)
    "IBCI.DE": 0.05,  # iShares Euro Inflation Linked Government Bond
    "EUNA.DE": 0.05,  # iShares Core Global Aggregate Bond, EUR hedged
    "XUTE.DE": 0.05,  # Xtrackers US Treasuries, EUR hedged (replaces 7USH.DE: longer history)
    "EUN5.DE": 0.05,  # iShares Core Euro Corporate Bond
    "EUNW.DE": 0.05,  # iShares Euro High Yield Corporate Bond
    "IS3C.DE": 0.05,  # iShares J.P. Morgan USD Emerging Markets Bond, EUR hedged
    "IUSP.DE": 0.05,  # iShares J.P. Morgan Emerging Markets Local Government Bond
    "4GLD.DE": 0.05,  # Xetra-Gold, physical gold (replaces PPFB.DE: longer history)
    "SXRS.DE": 0.05,  # iShares Diversified Commodity Swap
}


# The main settings dictionary that every other module will read from.
CONFIG = {
    "tickers": list(PORTFOLIO.keys()),    # just the ticker names, in order, as a list
    "weights": list(PORTFOLIO.values()),  # just the weights, in the same order, as a list
    "start_value": 1000,                  # EXAMPLE money invested on day one, in EUR
    "weekly_contribution": 25,            # EXAMPLE money added every week, in EUR
    "years": 10,                          # how many years into the future we simulate
    "n_sims": 10_000,                     # number of simulated futures (underscore is just for readability)
    "target": 20_000,                     # EXAMPLE final value in EUR we hope to reach
    "seed": 42,                           # fixed starting point for the random numbers, so results repeat
    # ---- Inflation and the savings-account alternative (all in the same currency) ----
    "inflation": 0.02,                    # assumed inflation per year (0.02 = 2%)
    "index_contributions": True,          # SWITCH: raise the weekly contribution over time? (False = flat)
    "contribution_growth": None,          # how fast per year; None = follow inflation, or e.g. 0.05 = 5%
    "index_every_weeks": 13,              # how often to raise it: 13 weeks = every quarter
    "savings_rate": 0.02,                 # EXAMPLE savings-account interest per year (0.02 = 2%)
    "target_in_todays_money": False,      # False: the target is in future (nominal) euros
    "report_in_todays_money": False,      # False: show everything in nominal euros; True: also today's money
    # ---- Price history ----
    "data_start": "2016-01-01",           # first date of price history to download (YYYY-MM-DD)
    "data_end": "2026-11-01",             # last date; fixed (not "today") so every run uses the same data
}


# Try to load the user's private numbers, which override the example values above.
# my_config.py is listed in .gitignore, so it never ends up on GitHub.
try:                                       # "try" means: attempt this, but don't crash if it fails
    from my_config import PRIVATE          # look for a file my_config.py with a dict called PRIVATE
    CONFIG.update(PRIVATE)                 # replace matching keys in CONFIG with the private values
    USING_PRIVATE = True                   # remember that private values are active
except ModuleNotFoundError:                # this error happens when my_config.py doesn't exist
    USING_PRIVATE = False                  # no private file? fine, keep the example values


def validate_config(config):
    """Check that the settings make sense. Raise a clear ValueError if they don't."""
    tickers = config["tickers"]  # grab the list of tickers from the dictionary
    weights = config["weights"]  # grab the list of weights from the dictionary

    # Every ticker needs exactly one weight.
    if len(tickers) != len(weights):
        raise ValueError(
            f"There are {len(tickers)} tickers but {len(weights)} weights; they must match."
        )

    # Weights can't be negative (that would mean short-selling, which we don't model).
    for ticker, weight in zip(tickers, weights):  # zip walks through both lists side by side
        if weight < 0:
            raise ValueError(f"Weight for {ticker} is {weight}; weights cannot be negative.")

    # Weights must add up to 1 (= 100%). We use isclose because decimals on a computer
    # are slightly imprecise: 0.1 + 0.2 gives 0.30000000000000004, not exactly 0.3.
    total = sum(weights)                                 # add up all the weights
    if not math.isclose(total, 1.0, abs_tol=1e-9):       # "close enough" to 1, within 0.000000001
        raise ValueError(f"Weights add up to {total:.6f}, but they must add up to 1.")

    # The remaining numbers must be sensible.
    if config["start_value"] < 0:
        raise ValueError("start_value cannot be negative.")
    if config["weekly_contribution"] < 0:
        raise ValueError("weekly_contribution cannot be negative.")
    if config["years"] <= 0:
        raise ValueError("years must be greater than 0.")
    if config["n_sims"] <= 0:
        raise ValueError("n_sims must be greater than 0.")
    if config["inflation"] <= -1:
        raise ValueError("inflation must be greater than -1 (= -100%).")
    if config["savings_rate"] <= -1:
        raise ValueError("savings_rate must be greater than -1 (= -100%).")
    growth = config["contribution_growth"]
    if growth is not None and growth <= -1:            # None is allowed: it means "follow inflation"
        raise ValueError("contribution_growth must be None or greater than -1 (= -100%).")
    if not isinstance(config["index_every_weeks"], int) or config["index_every_weeks"] < 1:
        raise ValueError("index_every_weeks must be a whole number of at least 1.")


# Run the check as soon as this file is imported, so a bad setting is caught immediately.
validate_config(CONFIG)


# This block only runs when you start this file directly (python -m src.config),
# not when another module imports it. Handy for a quick look at the settings.
if __name__ == "__main__":
    print(f"Number of ETFs:      {len(CONFIG['tickers'])}")       # how many tickers we have
    print(f"Sum of weights:      {sum(CONFIG['weights']):.4f}")   # should print 1.0000
    print(f"Start value:         {CONFIG['start_value']} EUR")    # shows private value if loaded
    print(f"Weekly contribution: {CONFIG['weekly_contribution']} EUR")
    print(f"Years:               {CONFIG['years']}")
    print(f"Simulations:         {CONFIG['n_sims']}")
    print(f"Target:              {CONFIG['target']} EUR")
    print(f"Inflation:           {CONFIG['inflation']:.1%} per year")
    print(f"Savings rate:        {CONFIG['savings_rate']:.1%} per year")
    print(f"Seed:                {CONFIG['seed']}")
    print("Config is valid.")                                     # we only get here if no error was raised
