"""Summarize simulated portfolio values into a few numbers that answer real questions."""

import numpy as np   # numpy: percentiles, means and other maths on arrays


def summarize(values, contributed, target):
    """Return a dict with the key results, all based on the FINAL value of each simulation.

    values       array (n_sims, n_weeks + 1) from run_portfolio
    contributed  total money put in yourself (start value + all contributions), in EUR
    target       the final value you hope to reach, in EUR

    Keys of the returned dict:
      p5, p50, p95  final value at the 5th, 50th (median) and 95th percentile, in EUR
      prob_target   share of simulations that end at or above the target (between 0 and 1)
      var_5         5% Value at Risk: contributed - p5. Positive = in the worst 5% of futures you lose
                    at least this much of your own money. Negative = even then you end above it.
      growth_share  share of the MEDIAN final value that is investment growth: (p50 - contributed) / p50
                    (the rest, 1 - growth_share, is your own money)
    """
    values = np.asarray(values)          # make sure we have a numpy array
    if values.ndim != 2:                 # ndim = number of dimensions; we expect a 2D table
        raise ValueError(f"values must be 2D (n_sims, n_weeks + 1), got {values.ndim}D.")

    final = values[:, -1]                # last column: the final value of every simulation

    # Three percentiles in one call; np.percentile sorts the values and reads off the positions.
    p5, p50, p95 = np.percentile(final, [5, 50, 95])

    # (final >= target) is an array of True/False; the mean of True/False is the share of True.
    prob_target = np.mean(final >= target)

    # Value at Risk relative to the money you put in (see the docstring for the sign).
    var_5 = contributed - p5

    # How much of a typical (median) final value is growth instead of your own money.
    growth_share = (p50 - contributed) / p50

    # float() turns numpy numbers into plain Python numbers, which print more cleanly.
    return {
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "prob_target": float(prob_target),
        "var_5": float(var_5),
        "growth_share": float(growth_share),
    }


def deflator(n_weeks, inflation):
    """Array of length n_weeks + 1: what 1 EUR at week t is worth in TODAY's money.

    With 2% inflation, 1 EUR after one year buys what 1 / 1.02 = 0.98 EUR buys today.
    """
    years = np.arange(n_weeks + 1) / 52        # time of each week in years: 0, 1/52, 2/52, ...
    return (1 + inflation) ** -years           # ** -years is the same as 1 / (1 + inflation) ** years


def to_todays_money(values, inflation):
    """Convert values over time (last axis = weeks, week 0 first) to today's money."""
    values = np.asarray(values)
    n_weeks = values.shape[-1] - 1             # shape[-1] = length of the LAST axis (works for 1D and 2D)
    return values * deflator(n_weeks, inflation)   # broadcasting: every row is multiplied week by week


def own_money(start_value, contributions, inflation=0.0):
    """Path of the money you put in yourself (length n_weeks + 1), optionally in today's money.

    Each contribution is converted with the inflation of the week it is paid (end of week t + 1),
    so the total measures the buying power of everything you put in.
    """
    contributions = np.asarray(contributions, dtype=float)
    d = deflator(len(contributions), inflation)          # d[0] = 1 for week 0, d[1] for week 1, ...
    paid = contributions * d[1:]                         # contribution of week t+1 in today's money
    return start_value + np.concatenate([[0.0], np.cumsum(paid)])   # running total, starting at week 0


def compare_to_savings(values, savings_values):
    """Opportunity cost: how does the portfolio do against the savings account?

    values          array (n_sims, n_weeks + 1): portfolio paths
    savings_values  array (n_weeks + 1,): the savings-account path (same money in)

    Returns a dict:
      savings_final  final value of the savings account
      prob_beat      share of simulations where the portfolio ends ABOVE the savings account
      extra_p5, extra_p50, extra_p95
                     portfolio final minus savings final, at the 5th / 50th / 95th percentile
                     (negative = you would have been better off with the savings account)
    """
    final = np.asarray(values)[:, -1]                    # final portfolio value of each simulation
    savings_final = float(savings_values[-1])            # one number: the savings account is risk-free
    extra = final - savings_final                        # extra gain (or loss) from investing
    p5, p50, p95 = np.percentile(extra, [5, 50, 95])
    return {
        "savings_final": savings_final,
        "prob_beat": float(np.mean(final > savings_final)),
        "extra_p5": float(p5),
        "extra_p50": float(p50),
        "extra_p95": float(p95),
    }


# Quick check: only runs with "python -m src.analyze".
if __name__ == "__main__":
    from src.config import CONFIG                            # settings (step 1)
    from src.data import download_prices                     # prices (step 2)
    from src.returns import log_returns, estimate_params     # parameters (step 3)
    from src.simulate import simulate_returns, run_portfolio # simulation (steps 4-6)

    # Run the whole chain: prices -> returns -> parameters -> simulated futures -> portfolio values.
    prices = download_prices(CONFIG["tickers"], CONFIG["data_start"], CONFIG["data_end"])
    mu, cov = estimate_params(log_returns(prices))
    n_weeks = CONFIG["years"] * 52
    sim = simulate_returns(mu, cov, n_weeks, CONFIG["n_sims"], CONFIG["seed"])
    values, contributed = run_portfolio(
        sim, CONFIG["weights"], CONFIG["start_value"], CONFIG["weekly_contribution"]
    )

    s = summarize(values, contributed, CONFIG["target"])     # the six numbers

    # Print them in plain language.
    # :<30 pads the label to 30 characters, :>10 right-aligns the number in 10, so columns line up.
    print(f"\n{'Money you put in:':<30}{contributed:>10,.0f} EUR")
    print(f"{'Final value, bad case (5%):':<30}{s['p5']:>10,.0f} EUR")
    print(f"{'Final value, median (50%):':<30}{s['p50']:>10,.0f} EUR")
    print(f"{'Final value, good case (95%):':<30}{s['p95']:>10,.0f} EUR")
    print(f"{'Chance to reach target:':<30}{s['prob_target']:>10.1%}")
    print(f"{'5% Value at Risk:':<30}{s['var_5']:>10,.0f} EUR")
    print(f"Median value: {s['growth_share']:.0%} growth, {1 - s['growth_share']:.0%} your own money")

    # The checks for this step.
    print(f"\nCheck p5 < p50 < p95:          {s['p5'] < s['p50'] < s['p95']}")
    print(f"Check 0 <= prob_target <= 1:   {0 <= s['prob_target'] <= 1}")
    print("(This quick check uses flat contributions and nominal money; main.py shows the full picture.)")
