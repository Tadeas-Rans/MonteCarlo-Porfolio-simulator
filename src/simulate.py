"""Simulate random future weekly returns (Monte Carlo) and turn them into portfolio values."""

import numpy as np   # numpy: fast maths on arrays and random number generation


def simulate_returns(mu, cov, n_weeks, n_sims, seed):
    """Return simulated weekly log returns, shape (n_sims, n_weeks, n_assets).

    Each week, the vector of returns is drawn from a multivariate normal N(mu, cov):
        returns = mu + L @ Z,   where Z ~ N(0, I) (independent) and L is the Cholesky factor (cov = L @ L.T).
    Multiplying by L turns independent random numbers into numbers with exactly covariance cov.
    """
    mu = np.asarray(mu, dtype=float)     # make sure mu is a numpy array (also works if a list is passed in)
    cov = np.asarray(cov, dtype=float)   # same for the covariance matrix
    n_assets = len(mu)                   # how many assets we are simulating

    # Guard: cov must be a square matrix with one row and one column per asset.
    if cov.shape != (n_assets, n_assets):
        raise ValueError(f"cov has shape {cov.shape}, expected ({n_assets}, {n_assets}).")

    # Cholesky decomposition: find the lower-triangular matrix L with L @ L.T == cov.
    # It only exists if cov is "positive definite" (no asset is an exact mix of the others).
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:        # numpy raises this when the decomposition is impossible
        raise ValueError(
            "Covariance matrix is not positive definite: two ETFs may be (almost) identical, "
            "or there are fewer weeks of data than ETFs."
        )

    # A random number generator with a fixed seed: same seed -> exactly the same "random" numbers.
    rng = np.random.default_rng(seed)

    # Draw independent standard normal numbers Z ~ N(0, 1): one per simulation, per week, per asset.
    z = rng.standard_normal((n_sims, n_weeks, n_assets))

    # Correlate them. z holds each week's numbers as a ROW vector, so instead of L @ z we use z @ L.T
    # (the same maths, transposed). The @ works on the last axis, for every simulation and week at once.
    correlated = z @ L.T

    # Shift to the right mean. "+=" changes the array in place, which saves memory (this array is large).
    correlated += mu
    return correlated


def run_portfolio(sim_returns, weights, start_value, weekly_contribution):
    """Turn simulated returns into portfolio values over time.

    Every week: 1) grow each holding, 2) add the contribution, 3) rebalance to the target weights.

    weekly_contribution can be ONE number (the same every week) or an array with one amount per
    week (length n_weeks), for example from contribution_schedule() when contributions are indexed.

    Returns:
      values            array (n_sims, n_weeks + 1): portfolio value in EUR; column 0 = start_value
      total_contributed the money you put in yourself: start_value + sum of all contributions
    """
    weights = np.asarray(weights, dtype=float)          # target weights as a numpy vector
    n_sims, n_weeks, n_assets = sim_returns.shape       # unpack the three sizes of the 3D array

    # Guard: one weight per asset.
    if len(weights) != n_assets:
        raise ValueError(f"Got {len(weights)} weights for {n_assets} assets.")

    # Turn the contribution into one amount per week. np.broadcast_to copies a single number
    # n_weeks times, and leaves an array that already has length n_weeks as it is.
    contributions = np.broadcast_to(np.asarray(weekly_contribution, dtype=float), (n_weeks,))

    # Empty table for the results: one row per simulation, one column per week (plus week 0).
    values = np.zeros((n_sims, n_weeks + 1))
    values[:, 0] = start_value                          # week 0: every simulation starts with start_value

    # Holdings in EUR per asset at the start: split the start value by the target weights.
    # start_value * weights has shape (n_assets,); np.tile copies that row for every simulation.
    holdings = np.tile(start_value * weights, (n_sims, 1))   # shape (n_sims, n_assets)

    for week in range(n_weeks):                         # week = 0, 1, ..., n_weeks - 1
        # 1) Grow: a log return r means the holding is multiplied by exp(r).
        holdings = holdings * np.exp(sim_returns[:, week, :])   # all simulations and assets at once

        # 2) Add this week's contribution to the total portfolio value.
        total = holdings.sum(axis=1) + contributions[week]      # sum across assets -> shape (n_sims,)

        # 3) Rebalance: split the new total over the assets by the target weights again.
        # total[:, None] turns shape (n_sims,) into (n_sims, 1) so it can multiply with weights (n_assets,).
        holdings = total[:, None] * weights

        values[:, week + 1] = total                     # store the value at the end of this week

    total_contributed = start_value + contributions.sum()   # your own money, no growth
    return values, float(total_contributed)


def contribution_schedule(weekly_contribution, n_weeks, inflation=0.0, index_every_weeks=13):
    """Return an array with the contribution for each week (length n_weeks).

    The amount is raised with inflation every `index_every_weeks` weeks (13 = every quarter).
    Week 1 to 13 pay the base amount; week 14 to 26 pay base x (1 + inflation)^(13/52); and so on.
    With inflation=0 every week pays the base amount.
    """
    weeks = np.arange(n_weeks)                          # 0, 1, 2, ..., n_weeks - 1
    periods = weeks // index_every_weeks                # // = whole-number division: 0,0,..,0,1,1,..
    years_passed = periods * index_every_weeks / 52     # time at the last raise, in years
    return weekly_contribution * (1 + inflation) ** years_passed


def run_savings(savings_rate, start_value, weekly_contribution, n_weeks):
    """Value of a savings account over time: same money in, fixed interest, no risk.

    Returns an array of length n_weeks + 1 (week 0 = start_value).
    Trick: a savings account is a "portfolio" of one asset with the same log return every week,
    so we reuse run_portfolio instead of writing (and testing) new bookkeeping code.
    """
    weekly_log_return = np.log(1 + savings_rate) / 52   # 52 of these add up to ln(1 + rate) per year
    sim = np.full((1, n_weeks, 1), weekly_log_return)   # 1 simulation, n_weeks weeks, 1 asset
    values, _ = run_portfolio(sim, [1.0], start_value, weekly_contribution)
    return values[0]                                    # the one path as a plain 1D array


# Quick check: only runs with "python -m src.simulate".
if __name__ == "__main__":
    import matplotlib.pyplot as plt                  # plotting library
    from src.config import CONFIG                    # settings (step 1)
    from src.data import download_prices             # prices (step 2)
    from src.returns import log_returns, estimate_params   # returns and parameters (step 3)

    n_weeks = CONFIG["years"] * 52                   # 10 years x 52 weeks = 520 weeks
    n_sims = CONFIG["n_sims"]                        # number of simulated futures
    start = CONFIG["start_value"]                    # starting amount in EUR
    contribution = CONFIG["weekly_contribution"]     # weekly amount in EUR
    weights = np.array(CONFIG["weights"])            # target weights of all ETFs

    # Historical data and parameters for ALL ETFs.
    prices = download_prices(CONFIG["tickers"], CONFIG["data_start"], CONFIG["data_end"])
    returns = log_returns(prices)
    mu, cov = estimate_params(returns)
    n_assets = len(mu)

    # ---- CHECK 1: zero returns -> final value must be exactly start + weeks x contribution ----
    zero_returns = np.zeros((5, n_weeks, n_assets))   # 5 simulations, all ETFs, every return exactly 0
    values_zero, contributed = run_portfolio(zero_returns, weights, start, contribution)
    expected = start + n_weeks * contribution         # the answer we know must come out
    print(f"\nCheck 1 - zero returns: final {values_zero[0, -1]:.2f}, expected {expected:.2f}, "
          f"passed: {np.allclose(values_zero[:, -1], expected)}")

    # ---- Simulate all ETFs together, with correlations (Cholesky) ----
    sim = simulate_returns(mu, cov, n_weeks, n_sims, CONFIG["seed"])
    print(f"Simulated returns shape: {sim.shape}  (n_sims, n_weeks, n_assets)")

    # ---- CHECK 2: simulated correlations should be close to the historical ones ----
    hist_corr = returns.corr().to_numpy()                             # 19 x 19 historical correlations
    # Stack all simulations and weeks into one long table: 5.2 million rows, one column per ETF.
    sim_corr = np.corrcoef(sim.reshape(-1, n_assets), rowvar=False)   # rowvar=False: columns are variables
    upper = np.triu_indices(n_assets, k=1)    # positions above the diagonal: every pair of ETFs once
    diff = np.abs(sim_corr[upper] - hist_corr[upper])                 # how far off each pair is
    print(f"Check 2 - correlations: {len(diff)} pairs, largest difference {diff.max():.4f}, "
          f"average difference {diff.mean():.4f}")

    # ---- The whole portfolio, WITH correlations ----
    values, contributed = run_portfolio(sim, weights, start, contribution)
    final = values[:, -1]                            # last column = final value of every simulation
    del sim                                          # free ~0.8 GB of memory; we don't need it any more

    # ---- The same, but pretending the ETFs are INDEPENDENT (only the diagonal of cov) ----
    cov_independent = np.diag(np.diag(cov))          # keep the variances, set every covariance to 0
    sim_ind = simulate_returns(mu, cov_independent, n_weeks, n_sims, CONFIG["seed"])
    final_ind = run_portfolio(sim_ind, weights, start, contribution)[0][:, -1]   # [0] = values
    del sim_ind                                      # free the memory again

    print(f"\nYour portfolio, {n_sims} simulations, {CONFIG['years']} years, "
          f"money put in {contributed:,.0f} EUR")
    print(f"{'':22} {'5%':>9} {'50%':>9} {'95%':>9}")
    for label, f in [("with correlations", final), ("pretend independent", final_ind)]:
        p5, p50, p95 = np.percentile(f, [5, 50, 95])     # three percentiles in one call
        print(f"{label:22} {p5:9,.0f} {p50:9,.0f} {p95:9,.0f}")

    # Chart: every pair of ETFs as one dot; x = historical, y = simulated correlation.
    # If the simulation is right, all dots lie on the diagonal line y = x.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([-1, 1], [-1, 1], color="grey", linewidth=1, linestyle="--")           # the y = x line
    ax.scatter(hist_corr[upper], sim_corr[upper], s=16, color="#2a6fdb", alpha=0.8)  # one dot per pair
    ax.set_xlim(-0.4, 1)
    ax.set_ylim(-0.4, 1)
    ax.set_xlabel("Historical correlation")
    ax.set_ylabel("Simulated correlation")
    ax.set_title(f"All {len(diff)} ETF pairs: simulated vs historical")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
