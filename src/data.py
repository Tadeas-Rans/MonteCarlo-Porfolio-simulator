"""Download weekly ETF prices from Yahoo Finance, cache them as CSV, and remove data spikes."""

from pathlib import Path   # Path makes working with folders and file names easy and safe

import numpy as np         # numpy: maths on arrays; used for logs and signs in the spike check
import pandas as pd        # pandas: tables of data (DataFrames); "pd" is the usual nickname
import yfinance as yf      # yfinance: downloads price data from Yahoo Finance; "yf" is its nickname


# The data/ folder sits next to src/, in the project root.
# __file__ is this file's own path; .parent goes up one folder (to src/), .parent again (to the root).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_prices(tickers, start, end):
    """Return a DataFrame of weekly adjusted close prices.

    Rows are weeks (dates), columns are tickers, in the same order as `tickers`.
    Only weeks where EVERY ticker has a price are kept, and one-week data spikes are removed.
    """
    # Build the cache file name from the dates, e.g. data/prices_2016-01-01_2026-11-01.csv
    cache_file = DATA_DIR / f"prices_{start}_{end}.csv"

    # 1) If we downloaded this before, read the saved file instead of downloading again.
    prices = None                                                        # "nothing loaded yet"
    if cache_file.exists():                                              # does the file already exist?
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)  # first column = dates
        if all(ticker in cached.columns for ticker in tickers):          # does it contain every ticker?
            print(f"Loaded prices from cache: {cache_file.name}")
            prices = cached[tickers]                                     # take the columns in OUR order
        else:
            print("Cache is missing some tickers, downloading again.")

    # 2) No usable cache? Download from Yahoo Finance.
    if prices is None:
        print(f"Downloading weekly prices for {len(tickers)} tickers...")
        raw = yf.download(
            tickers,              # list of ticker symbols
            start=start,          # first date
            end=end,              # last date
            interval="1wk",       # one price per week instead of per day
            auto_adjust=True,     # adjust prices for dividends and splits, so "Close" = adjusted close
            progress=False,       # don't print a progress bar
        )
        prices = raw["Close"]     # yfinance returns Open/High/Low/Close/Volume; we only want Close

        # Check that every ticker actually came back with data.
        bad = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]  # missing or empty
        if bad:                                                         # a non-empty list counts as True
            raise ValueError(f"No price data for: {bad}. Check the ticker symbols on Yahoo Finance.")

        # Put the columns in our order (Yahoo sorts them alphabetically; the weights depend on our order).
        prices = prices[tickers]

        # Keep only weeks where all ETFs have a price. This cuts off the early years before the
        # youngest ETF existed, so every column covers exactly the same weeks.
        prices = prices.dropna()

        # Save the RAW prices (before spike removal) so we can change the cleaning rule later
        # without downloading again.
        DATA_DIR.mkdir(exist_ok=True)   # create data/ if it doesn't exist yet (no error if it does)
        prices.to_csv(cache_file)       # write the table to a CSV file
        print(f"Saved prices to cache: {cache_file.name}")

    # 3) Clean the data (both for cached and freshly downloaded prices).
    return remove_spikes(prices)


def trim_history(prices, years):
    """Keep only the last `years` years of prices. None keeps everything.

    If there is less history than asked for (because the youngest ETF is too new),
    all of it is kept and a warning is printed, so the user knows the real window.
    """
    if years is None:                                     # None = no limit, use every week we have
        return prices
    available = (prices.index[-1] - prices.index[0]).days / 365.25   # years between first and last week
    if available < years:                                 # asked for more than exists?
        print(f"Warning: asked for {years} years of history, but all ETFs together only have "
              f"{available:.1f} years (from {prices.index[0].date()}). Using all of it.")
        return prices
    # pd.DateOffset(years=...) counts calendar years back, so leap years are handled for us.
    # It needs a whole number, so a fraction like 7.5 years is converted to months first.
    cutoff = prices.index[-1] - pd.DateOffset(months=round(years * 12))
    return prices[prices.index >= cutoff]                 # keep the rows on or after the cutoff date


def remove_spikes(prices, threshold=5.0):
    """Find one-week data errors ("spikes") and replace them by interpolation.

    A price counts as a spike when ALL four conditions hold:
      1. the move INTO that week is huge for this ETF (more than `threshold` x its normal weekly move),
      2. the move OUT of it the next week is also huge,
      3. the two moves go in opposite directions and mostly cancel out (price jumps and comes back),
      4. the rest of the market was calm in both weeks (a real crash moves many ETFs at once).
    Works best with several tickers: with only one, condition 4 can never be met, so nothing is removed.
    """
    returns = np.log(prices / prices.shift(1))    # weekly log returns (first row is NaN)

    # "Normal weekly move" per ETF, measured robustly with the MAD (median absolute deviation).
    # Unlike the standard deviation, the MAD is barely affected by the outliers we're looking for.
    # Multiplying by 1.4826 makes it comparable to a standard deviation for normal data.
    typical_move = 1.4826 * (returns - returns.median()).abs().median()

    z = returns / typical_move           # each return measured in "normal moves" (like a z-score)
    z_next = z.shift(-1)                 # next week's z, lined up on the same row (shift UP by one)

    big_in = z.abs() > threshold                          # condition 1: huge move into this week
    big_out = z_next.abs() > threshold                    # condition 2: huge move out the next week
    reverses = np.sign(z) != np.sign(z_next)              # condition 3a: opposite directions
    cancels = (z + z_next).abs() < 0.5 * z.abs()          # condition 3b: at least half of the jump undone

    market = z.abs().median(axis=1)                       # how wild the typical ETF was each week
    market_both_weeks = np.maximum(market, market.shift(-1))   # the wilder of this week and next
    calm = market_both_weeks < 2                          # condition 4: market calm in both weeks

    # Combine: & means "and". calm is one value per week, so line it up with every column.
    is_spike = big_in & big_out & reverses & cancels
    is_spike = is_spike.mul(calm, axis=0).astype(bool)    # keep a spike only in calm weeks

    # Report what we found, so nothing is changed silently.
    for ticker in prices.columns:
        for date in prices.index[is_spike[ticker]]:      # dates where this ticker has a spike
            print(f"Spike removed: {ticker} on {date.date()} (price {prices.loc[date, ticker]:.2f})")

    # Replace spikes by NaN, then fill the gap with a straight line between the neighbouring weeks.
    cleaned = prices.mask(is_spike)        # mask() puts NaN wherever is_spike is True
    return cleaned.interpolate()           # interpolate() draws a straight line over the NaN gaps


# Quick check: only runs with "python -m src.data", not when another module imports this file.
if __name__ == "__main__":
    import matplotlib.pyplot as plt   # plotting library; imported here because only the check needs it
    from src.config import CONFIG     # our settings from step 1

    # Get the prices using the tickers and dates from the config.
    prices = download_prices(CONFIG["tickers"], CONFIG["data_start"], CONFIG["data_end"])

    # Print a short summary of what we got.
    n_weeks = len(prices)                                     # number of rows = number of weeks
    print(f"Shape: {prices.shape}  (weeks, tickers)")         # .shape gives (rows, columns)
    print(f"From {prices.index[0].date()} to {prices.index[-1].date()}")  # first and last date
    print(f"That is {n_weeks} weeks = {n_weeks / 52:.1f} years")          # 52 weeks per year
    print(f"Empty values left: {prices.isna().sum().sum()}")  # count missing values; should be 0
    print("\nLast 3 weeks (first 4 ETFs):")
    print(prices.iloc[-3:, :4].round(2))                      # last 3 rows, first 4 columns

    # Divide every price by its first price, so each ETF starts at 1.0.
    # Now all ETFs are on the same scale and you can compare growth directly.
    growth = prices / prices.iloc[0]

    # 19 lines on one chart would be unreadable spaghetti, so draw one small chart per ETF.
    fig, axes = plt.subplots(4, 5, figsize=(15, 9), sharex=True, sharey=True)  # 4 rows x 5 columns grid
    for ax, ticker in zip(axes.flat, CONFIG["tickers"]):      # .flat walks through the grid one by one
        ax.plot(growth.index, growth[ticker], color="#2a6fdb", linewidth=1.5)  # the price line
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")          # reference line at the start value
        ax.set_title(ticker, fontsize=10)                     # ETF name above each small chart
        ax.grid(alpha=0.3)                                    # faint grid lines
    for ax in list(axes.flat)[len(CONFIG["tickers"]):]:       # any grid slots left over after the last ETF
        ax.axis("off")                                        # hide the empty ones
    fig.suptitle("Growth of 1 EUR invested (adjusted close, weekly)")  # one title for the whole figure
    fig.autofmt_xdate()                                       # tilt the date labels so they don't overlap
    fig.tight_layout()                                        # stop titles and labels from overlapping
    plt.show()                                                # open the chart window
