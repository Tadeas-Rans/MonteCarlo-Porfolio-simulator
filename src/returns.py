"""Turn prices into weekly log returns and estimate their mean and covariance."""

import numpy as np   # numpy: fast maths on arrays (vectors and matrices); "np" is the usual nickname


def log_returns(prices):
    """Return a DataFrame of weekly log returns: ln(price this week / price last week).

    Same columns as `prices`, one row fewer (the first week has no previous week).
    """
    # prices.shift(1) moves every column down by one row, so each row lines up with LAST week's price.
    # Dividing gives the growth factor per week (e.g. 1.02 = +2%); np.log takes the natural log of it.
    returns = np.log(prices / prices.shift(1))

    # The first row is empty (NaN) because week 1 has no previous week; drop it.
    return returns.dropna()


def estimate_params(returns):
    """Return (mu, cov) estimated from historical weekly log returns.

    mu  = numpy vector, length n_assets: the average weekly log return of each ETF.
    cov = numpy matrix, n_assets x n_assets: the weekly covariance between every pair of ETFs.
    """
    mu = returns.mean().to_numpy()   # average of each column; .to_numpy() turns pandas into a plain numpy array
    cov = returns.cov().to_numpy()   # sample covariance matrix (divides by n-1, like in statistics class)
    return mu, cov


# Quick check: only runs with "python -m src.returns".
if __name__ == "__main__":
    from src.config import CONFIG              # settings from step 1
    from src.data import download_prices       # price download from step 2

    # Load prices (from the cache) and turn them into returns.
    prices = download_prices(CONFIG["tickers"], CONFIG["data_start"], CONFIG["data_end"])
    returns = log_returns(prices)
    mu, cov = estimate_params(returns)

    print(f"Returns shape: {returns.shape}  (weeks, tickers)")   # one row fewer than prices
    print(f"mu shape:      {mu.shape}")                           # (19,)  -> a vector
    print(f"cov shape:     {cov.shape}")                          # (19, 19) -> a square matrix

    # Per-ETF table. Weekly numbers are tiny, so also show them per year:
    #   yearly mean ~ weekly mean x 52,   yearly volatility ~ weekly st.dev. x sqrt(52).
    vol = np.sqrt(np.diag(cov))                 # the diagonal of cov holds the variances; sqrt gives st.dev.
    print(f"\n{'ETF':<9} {'mean/wk':>8} {'mean/yr':>8} {'vol/yr':>7}")   # table header
    for ticker, m, v in zip(CONFIG["tickers"], mu, vol):             # walk through the three lists together
        print(f"{ticker:<9} {m:8.4f} {m * 52:8.1%} {v * np.sqrt(52):7.1%}")

    # The portfolio as a whole: weighted average of the means (weights @ mu is a dot product).
    weights = np.array(CONFIG["weights"])       # turn the list of weights into a numpy vector
    print(f"\nPortfolio mean weekly log return: {weights @ mu:.4f}")

    # The two checks for this step.
    print(f"cov is square:    {cov.shape[0] == cov.shape[1]}")       # same number of rows and columns
    print(f"cov is symmetric: {np.allclose(cov, cov.T)}")            # cov equals its own transpose (.T)
