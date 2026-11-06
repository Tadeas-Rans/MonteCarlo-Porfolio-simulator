"""Charts of the simulated portfolio values."""

import matplotlib.pyplot as plt   # plotting library; "plt" is the usual nickname
import numpy as np                # numpy: percentiles and other maths on arrays

BLUE = "#2a6fdb"     # one colour for everything about the simulation, in different strengths
INK = "#333333"      # dark grey for reference lines (target, own money)
ORANGE = "#d9730d"   # the savings-account alternative: a clearly different colour from the portfolio


def fan_chart(values, target=None, own_money=None, savings=None, title="Simulated portfolio value over time"):
    """Fan chart: median path with 25-75% and 5-95% bands over time. Returns the figure.

    values     array (n_sims, n_weeks + 1) from run_portfolio
    target     optional: draw a horizontal line at this value (EUR)
    own_money  optional: path (length n_weeks + 1) of the money you put in yourself
    savings    optional: path (length n_weeks + 1) of the savings-account alternative
    title      chart title (e.g. to say the values are in today's money)
    """
    n_weeks = values.shape[1] - 1                 # number of columns minus week 0
    years = np.arange(n_weeks + 1) / 52           # x-axis in years instead of weeks: easier to read

    # For EVERY week (axis=0 = across the simulations), the 5 percentiles we need.
    # Result shape (5, n_weeks + 1): row 0 = 5% path, row 1 = 25% path, ..., row 4 = 95% path.
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95], axis=0)

    fig, ax = plt.subplots(figsize=(10, 5.5))                     # one chart
    ax.fill_between(years, p5, p95, color=BLUE, alpha=0.18, linewidth=0,
                    label="5-95% of futures")                     # light band: middle 90%
    ax.fill_between(years, p25, p75, color=BLUE, alpha=0.38, linewidth=0,
                    label="25-75% of futures")                    # darker band: middle 50%
    ax.plot(years, p50, color=BLUE, linewidth=2, label="Median")  # the typical future

    if savings is not None:                                       # only if the caller gave it
        ax.plot(years, savings, color=ORANGE, linewidth=2, label="Savings account instead")
    if own_money is not None:
        ax.plot(years, own_money, color=INK, linewidth=1.5, label="Money you put in")
    if target is not None:
        ax.axhline(target, color=INK, linewidth=1, linestyle="--",
                   label=f"Target {target:,.0f} EUR")             # horizontal dashed line

    ax.set_title(title)
    ax.set_xlabel("Years from now")
    ax.set_ylabel("Portfolio value (EUR)")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}")        # 25000 -> "25,000" on the axis
    ax.set_xlim(0, years[-1])                                     # no empty space left and right
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def final_hist(values, target=None, contributed=None, savings_final=None, title=None):
    """Histogram of the final values, with the 5th / 50th / 95th percentile marked. Returns the figure.

    values         array (n_sims, n_weeks + 1) from run_portfolio
    target         optional: draw a vertical line at this value (EUR)
    contributed    optional: draw a vertical line at the money you put in (EUR)
    savings_final  optional: draw a vertical line at the savings account's final value (EUR)
    title          optional chart title
    """
    final = values[:, -1]                                  # last column: final value of each simulation
    p5, p50, p95 = np.percentile(final, [5, 50, 95])       # the three key percentiles

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(final, bins=60, color=BLUE, alpha=0.6, edgecolor="white", linewidth=0.5)  # 60 bars

    # Percentile lines in the simulation colour; reference lines in dark grey.
    ax.axvline(p5, color=BLUE, linewidth=1.2, linestyle=":", label=f"5%: {p5:,.0f}")
    ax.axvline(p50, color=BLUE, linewidth=2, label=f"Median: {p50:,.0f}")
    ax.axvline(p95, color=BLUE, linewidth=1.2, linestyle=":", label=f"95%: {p95:,.0f}")
    if contributed is not None:
        ax.axvline(contributed, color=INK, linewidth=1.5, label=f"Money put in: {contributed:,.0f}")
    if savings_final is not None:
        ax.axvline(savings_final, color=ORANGE, linewidth=2, label=f"Savings account: {savings_final:,.0f}")
    if target is not None:
        ax.axvline(target, color=INK, linewidth=1, linestyle="--", label=f"Target: {target:,.0f}")

    if title is None:                                             # default title if none was given
        title = f"Final value after {(values.shape[1] - 1) / 52:.0f} years ({len(final):,} simulations)"
    ax.set_title(title)
    ax.set_xlabel("Final portfolio value (EUR)")
    ax.set_ylabel("Number of simulations")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")        # 25000 -> "25,000" on the axis
    ax.grid(alpha=0.3, axis="y")                                  # horizontal grid lines only
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# Quick check: only runs with "python -m src.plots".
if __name__ == "__main__":
    from src.config import CONFIG                            # settings (step 1)
    from src.data import download_prices                     # prices (step 2)
    from src.returns import log_returns, estimate_params     # parameters (step 3)
    from src.simulate import simulate_returns, run_portfolio # simulation (steps 4-6)

    # The whole chain again: prices -> parameters -> simulated portfolio values.
    prices = download_prices(CONFIG["tickers"], CONFIG["data_start"], CONFIG["data_end"])
    mu, cov = estimate_params(log_returns(prices))
    sim = simulate_returns(mu, cov, CONFIG["years"] * 52, CONFIG["n_sims"], CONFIG["seed"])
    values, contributed = run_portfolio(
        sim, CONFIG["weights"], CONFIG["start_value"], CONFIG["weekly_contribution"]
    )

    # The checks for this step, in numbers.
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95], axis=0)
    width_1y = p95[52] - p5[52]           # width of the 5-95% band after 1 year (week 52)
    width_end = p95[-1] - p5[-1]          # width at the end
    print(f"Band width after 1 year: {width_1y:,.0f} EUR, at the end: {width_end:,.0f} EUR")
    print(f"Check fan widens:        {width_end > width_1y}")
    print(f"Check median inside bands every week: {np.all((p5 <= p25) & (p25 <= p50) & (p50 <= p75) & (p75 <= p95))}")

    fan_chart(values, target=CONFIG["target"])                            # figure 1 (main.py adds more lines)
    final_hist(values, target=CONFIG["target"], contributed=contributed)  # figure 2
    plt.show()                                                            # open both windows
