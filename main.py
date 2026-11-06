"""Portfolio Monte Carlo simulator: run everything with one command.

Usage (from the project folder, with the portfolio-sim environment active):
    python main.py
"""

import time                       # to measure how long the run takes
from pathlib import Path          # for the figures/ folder

import matplotlib.pyplot as plt   # to show the charts at the end

from src.analyze import summarize                          # step 7: the six key numbers
from src.config import CONFIG, USING_PRIVATE               # step 1: settings (already validated on import)
from src.data import download_prices                       # step 2: prices
from src.plots import fan_chart, final_hist                # step 8: charts
from src.returns import estimate_params, log_returns       # step 3: returns, mu and cov
from src.simulate import run_portfolio, simulate_returns   # steps 4-6: Monte Carlo

FIGURES_DIR = Path(__file__).resolve().parent / "figures"  # charts are saved here (in .gitignore)


def main():
    """Run the full simulation and print and plot the results."""
    t0 = time.time()                                       # start the stopwatch

    # ---- 1. Settings ----
    n_weeks = CONFIG["years"] * 52                         # horizon in weeks
    source = "PRIVATE (my_config.py)" if USING_PRIVATE else "EXAMPLE (src/config.py)"
    print(f"Settings: {source}")
    print(f"  {len(CONFIG['tickers'])} ETFs, start {CONFIG['start_value']:,} EUR, "
          f"{CONFIG['weekly_contribution']:,} EUR/week, {CONFIG['years']} years, "
          f"{CONFIG['n_sims']:,} simulations, target {CONFIG['target']:,} EUR")

    # ---- 2. Data ----
    prices = download_prices(CONFIG["tickers"], CONFIG["data_start"], CONFIG["data_end"])
    print(f"Data: {len(prices)} weeks, {prices.index[0].date()} to {prices.index[-1].date()}")

    # ---- 3. Returns and parameters ----
    returns = log_returns(prices)
    mu, cov = estimate_params(returns)

    # ---- 4. Simulate ----
    sim = simulate_returns(mu, cov, n_weeks, CONFIG["n_sims"], CONFIG["seed"])
    values, contributed = run_portfolio(
        sim, CONFIG["weights"], CONFIG["start_value"], CONFIG["weekly_contribution"]
    )
    del sim                                                # free ~0.8 GB; only `values` is needed now

    # ---- 5. Analyze ----
    s = summarize(values, contributed, CONFIG["target"])

    # ---- 6. Print the summary ----
    print("\n" + "=" * 46)                                 # a line of 46 "=" signs
    print(f" RESULTS after {CONFIG['years']} years")
    print("=" * 46)
    print(f"{'Money you put in:':<30}{contributed:>10,.0f} EUR")
    print(f"{'Final value, bad case (5%):':<30}{s['p5']:>10,.0f} EUR")
    print(f"{'Final value, median (50%):':<30}{s['p50']:>10,.0f} EUR")
    print(f"{'Final value, good case (95%):':<30}{s['p95']:>10,.0f} EUR")
    print(f"{'Chance to reach target:':<30}{s['prob_target']:>10.1%}")
    print(f"{'5% Value at Risk:':<30}{s['var_5']:>10,.0f} EUR")
    print(f"{'Median: growth / own money:':<30}{s['growth_share']:>9.0%} / {1 - s['growth_share']:.0%}")
    print("=" * 46)
    print("Assumes normal returns from past data; no fees, taxes or crashes beyond history.")

    # ---- 7. Charts: save to figures/ and show ----
    FIGURES_DIR.mkdir(exist_ok=True)                       # create figures/ if needed
    fan = fan_chart(values, target=CONFIG["target"], contributed=contributed)
    hist = final_hist(values, target=CONFIG["target"], contributed=contributed)
    fan.savefig(FIGURES_DIR / "fan_chart.png", dpi=150)    # dpi = resolution (dots per inch)
    hist.savefig(FIGURES_DIR / "final_hist.png", dpi=150)
    print(f"\nCharts saved to {FIGURES_DIR.name}/  (finished in {time.time() - t0:.1f} s)")
    plt.show()                                             # open the chart windows; close them to exit


# Run main() only when this file is started directly, not when it is imported (e.g. by tests).
if __name__ == "__main__":
    main()
