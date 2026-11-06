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
