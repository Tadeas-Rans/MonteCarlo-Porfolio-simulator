# Portfolio Monte Carlo Simulator

How might an ETF portfolio with a fixed weekly contribution grow over the next 10 years, and how likely is it to reach a target value?

This tool downloads weekly prices for a set of EUR-listed ETFs, estimates their returns and correlations, and simulates 10,000 possible futures with the weekly contribution added. It then reports the range of outcomes.

![Fan chart: simulated portfolio value over 10 years](docs/fan_chart_example.png)

*Example inputs: €1,000 start, €25 per week, 19 ETFs, 10 years, 10,000 simulations.*

## Method

1. **Data.** Weekly adjusted close prices from Yahoo Finance (`yfinance`), cached as CSV. Only weeks where every ETF has a price are kept. One-week data errors (a price that jumps and immediately comes back while the rest of the market is calm) are detected with a robust z-score (MAD) and interpolated.
2. **Log returns.** `r_t = ln(P_t / P_{t-1})`. Log returns add up over time, so a multi-week return is just a sum.
3. **Parameters.** The mean vector μ and the sample covariance matrix Σ of the weekly log returns.
4. **Correlated simulation.** Each week's return vector is drawn as `μ + L·z`, with `z ~ N(0, I)` and `L` the Cholesky factor of Σ (`Σ = L·Lᵀ`). This keeps the historical correlations between ETFs. Ignoring them would make the risk look about 3 times smaller for this portfolio.
5. **Portfolio with contributions.** Every week the holdings grow, the contribution is added, and the portfolio is rebalanced to the target weights.
6. **Summary.** The 5th, 50th and 95th percentiles of the final value; the probability of reaching the target; the 5% Value at Risk (money put in minus the 5th percentile); and the share of the median final value that comes from investment growth.

## Result (example inputs)

| | |
|---|---|
| Money put in | €14,000 |
| Final value: 5% / median / 95% | €15,350 / €19,887 / €25,873 |
| Chance to reach €20,000 | 48.6% |
| 5% Value at Risk | −€1,350 (even the worst 5% of outcomes end above the money put in) |
| Median final value | 30% growth, 70% own contributions |

Over 10 years, the amount you save matters more than the returns: most of a typical final value is your own money.

## Limitations

- **The past is not the future.** The parameters come from about 8.5 years of data (2018–2026), mostly a good period for stocks and gold. The means in particular are estimated very imprecisely.
- **Normal distribution: no fat tails.** Real markets have sudden crashes (like March 2020) far more often than a normal distribution predicts, so the real downside is probably larger.
- **Constant parameters.** Volatility and correlations are assumed to stay the same for 10 years. In reality correlations tend to rise in a crisis.
- **No fees, taxes or trading costs.** ETF fees and Dutch box 3 tax would lower every outcome.
- **Weekly rebalancing** is assumed, with no costs. Contributions are added at the end of each week.

## How to run

```bash
conda create -n portfolio-sim python=3.12 -y
conda activate portfolio-sim
pip install -r requirements.txt

python main.py      # run the full simulation, print the summary, show the charts
pytest -v           # run the tests
```

The settings are in `src/config.py`. To use your own amounts without committing them, create `my_config.py` in the project root. It is listed in `.gitignore`.

```python
PRIVATE = {"start_value": 5000, "weekly_contribution": 50, "target": 50_000}
```

## Project structure

```
main.py              runs everything in order
src/config.py        settings and validation
src/data.py          download, cache and clean prices
src/returns.py       log returns, mean and covariance
src/simulate.py      Cholesky Monte Carlo and the portfolio with contributions
src/analyze.py       percentiles, probability, VaR, growth share
src/plots.py         fan chart and histogram
tests/               pytest tests with known answers
```

## Planned

A bootstrap method next to the normal distribution (fat tails), rebalancing on/off, maximum drawdown, USD→EUR conversion.

## Note on AI use

AI (Claude) was used as a tutor and starting point for this project. The final code is my own rewrite, and I can explain every line.
