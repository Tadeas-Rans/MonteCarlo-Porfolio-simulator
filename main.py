"""Portfolio Monte Carlo simulator: run everything with one command.

Usage (from the project folder, with the portfolio-sim environment active):
    python main.py
"""

import time                       # to measure how long the run takes
from pathlib import Path          # for the figures/ folder

import matplotlib.pyplot as plt   # to show the charts at the end

from src.analyze import compare_to_savings, own_money, summarize, to_todays_money   # step 7 + 12
from src.config import CONFIG, USING_PRIVATE               # step 1: settings (already validated on import)
from src.data import download_prices                       # step 2: prices
from src.plots import fan_chart, final_hist                # step 8: charts
from src.returns import estimate_params, log_returns       # step 3: returns, mu and cov
from src.simulate import (                                 # steps 4-6 + 12: Monte Carlo and savings
    contribution_schedule,
    run_portfolio,
    run_savings,
    simulate_returns,
)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"  # charts are saved here (in .gitignore)


def row(label, nominal, real=None, fmt="{:>12,.0f}"):
    """Print one line of the results table: a label, the nominal value and (optionally) today's money."""
    text = f"{label:<32}{fmt.format(nominal)}"     # label padded to 32 characters, then the number
    if real is not None:                            # second column only when today's money is shown
        text += fmt.format(real)
    print(text)


def main():
    """Run the full simulation and print and plot the results."""
    t0 = time.time()                                       # start the stopwatch
    c = CONFIG                                             # short name, to keep lines readable

    # ---- 1. Settings ----
    n_weeks = c["years"] * 52                              # horizon in weeks
    infl = c["inflation"]                                  # inflation per year
    source = "PRIVATE (my_config.py)" if USING_PRIVATE else "EXAMPLE (src/config.py)"
    print(f"Settings: {source}")
    print(f"  {len(c['tickers'])} ETFs, start {c['start_value']:,} EUR, "
          f"{c['weekly_contribution']:,} EUR/week, {c['years']} years, {c['n_sims']:,} simulations")
    print(f"  inflation {infl:.1%}/yr, savings rate {c['savings_rate']:.1%}/yr")

    # ---- 2. Data, returns and parameters ----
    prices = download_prices(c["tickers"], c["data_start"], c["data_end"])
    print(f"Data: {len(prices)} weeks, {prices.index[0].date()} to {prices.index[-1].date()}")
    mu, cov = estimate_params(log_returns(prices))

    # ---- 3. Contributions: one amount per week, raised over time if switched on ----
    if not c["index_contributions"]:                       # switch off: the same amount every week
        index_rate = 0.0
    elif c["contribution_growth"] is None:                 # switch on, no own rate: follow inflation
        index_rate = infl
    else:                                                  # switch on with an own growth rate
        index_rate = c["contribution_growth"]
    contributions = contribution_schedule(c["weekly_contribution"], n_weeks, index_rate, c["index_every_weeks"])
    if index_rate == 0:
        print("  contributions: flat, the same amount every week")
    else:
        print(f"  contributions: +{index_rate:.1%}/yr, raised every {c['index_every_weeks']} weeks "
              f"({contributions[0]:,.2f} -> {contributions[-1]:,.2f} EUR/week)")

    # ---- 4. Simulate the portfolio, and the savings account with the same money ----
    sim = simulate_returns(mu, cov, n_weeks, c["n_sims"], c["seed"])
    values, contributed = run_portfolio(sim, c["weights"], c["start_value"], contributions)
    del sim                                                # free ~0.8 GB; only `values` is needed now
    savings = run_savings(c["savings_rate"], c["start_value"], contributions, n_weeks)

    # ---- 5. The same results in today's money ----
    values_real = to_todays_money(values, infl)
    savings_real = to_todays_money(savings, infl)
    own_real = own_money(c["start_value"], contributions, infl)       # your money, in today's buying power

    # The target in both units: (1 + inflation)^years converts between today and the end.
    price_rise = (1 + infl) ** c["years"]
    if c["target_in_todays_money"]:
        target_real, target_nominal = c["target"], c["target"] * price_rise
    else:
        target_nominal, target_real = c["target"], c["target"] / price_rise

    # ---- 6. Analyze ----
    s_nom = summarize(values, contributed, target_nominal)
    s_real = summarize(values_real, own_real[-1], target_real)
    cmp_nom = compare_to_savings(values, savings)
    cmp_real = compare_to_savings(values_real, savings_real)

    # ---- 7. Print the results table ----
    # show_real switches the second column ("today's money") on or off; nominal is always shown.
    show_real = c["report_in_todays_money"]
    def real_or_none(x):                                   # helper: the value if shown, else None
        return x if show_real else None

    line = "=" * 56
    header = f"{'Nominal':>21}" + (f"{'Today':>12}" if show_real else "")
    print(f"\n{line}\n RESULTS after {c['years']} years{header}\n{line}")
    row("Money you put in", contributed, real_or_none(own_real[-1]))
    row("Final value, bad case (5%)", s_nom["p5"], real_or_none(s_real["p5"]))
    row("Final value, median (50%)", s_nom["p50"], real_or_none(s_real["p50"]))
    row("Final value, good case (95%)", s_nom["p95"], real_or_none(s_real["p95"]))
    row("Target", target_nominal, real_or_none(target_real))
    print(f"{'Chance to reach target':<32}{s_nom['prob_target']:>12.1%}")
    row("5% Value at Risk", s_nom["var_5"], real_or_none(s_real["var_5"]))
    row("Median: share from growth", s_nom["growth_share"], real_or_none(s_real["growth_share"]), "{:>12.0%}")
    print(f"{line}\n OPPORTUNITY COST vs savings account at {c['savings_rate']:.1%}/yr\n{line}")
    row("Savings account final value", cmp_nom["savings_final"], real_or_none(cmp_real["savings_final"]))
    print(f"{'Chance portfolio beats savings':<32}{cmp_nom['prob_beat']:>12.1%}")
    row("Extra from investing, bad (5%)", cmp_nom["extra_p5"], real_or_none(cmp_real["extra_p5"]))
    row("Extra from investing, median", cmp_nom["extra_p50"], real_or_none(cmp_real["extra_p50"]))
    row("Extra from investing, good (95%)", cmp_nom["extra_p95"], real_or_none(cmp_real["extra_p95"]))
    print(line)
    print("Assumes normal returns from past data, fixed inflation and savings rate;")
    print("no fees, taxes or crashes beyond history.")

    # ---- 8. Charts (nominal, or today's money if switched on): save to figures/ and show ----
    if show_real:                                          # pick which numbers go into the charts
        v, own, sav, tgt, unit = values_real, own_real, savings_real, target_real, "today's money"
    else:
        own_nominal = own_money(c["start_value"], contributions)      # inflation 0 = plain euros
        v, own, sav, tgt, unit = values, own_nominal, savings, target_nominal, "nominal EUR"
    FIGURES_DIR.mkdir(exist_ok=True)                       # create figures/ if needed
    fan = fan_chart(v, target=tgt, own_money=own, savings=sav,
                    title=f"Simulated portfolio value over time ({unit})")
    hist = final_hist(v, target=tgt, contributed=own[-1], savings_final=sav[-1],
                      title=f"Final value after {c['years']} years ({unit})")
    fan.savefig(FIGURES_DIR / "fan_chart.png", dpi=150)    # dpi = resolution (dots per inch)
    hist.savefig(FIGURES_DIR / "final_hist.png", dpi=150)
    print(f"\nCharts saved to {FIGURES_DIR.name}/  (finished in {time.time() - t0:.1f} s)")
    plt.show()                                             # open the chart windows; close them to exit


# Run main() only when this file is started directly, not when it is imported (e.g. by tests).
if __name__ == "__main__":
    main()
