# European Option Pricing: Black–Scholes and Monte Carlo

This project develops and validates two complementary approaches to pricing European options: the analytical Black–Scholes model and a Monte Carlo estimator. It uses the analytical solution as a benchmark for studying simulation error, convergence, confidence intervals, and antithetic variates.

The emphasis is on numerical reasoning and reproducible results. Reusable pricing functions live in Python modules; the notebooks use those functions to investigate model behavior and communicate the findings.

## Project motivation

Option pricing provides a compact setting for connecting probability theory, stochastic models, numerical methods, and software validation. Black–Scholes gives a closed-form reference value under specific assumptions. Monte Carlo simulation approaches the same value by sampling risk-neutral outcomes, making its statistical error visible and measurable.

The project addresses three practical questions:

1. Do the analytical prices and risk sensitivities agree with established relationships and numerical checks?
2. How does Monte Carlo estimation error change as the path count grows?
3. Can antithetic variates reduce estimator variance for a fixed simulation budget?

## Project overview

The code supports European calls and puts, continuous dividend yield, price sensitivities, direct simulation of terminal asset prices, Monte Carlo estimates with confidence intervals, and antithetic sampling. Automated tests cover pricing relationships, limiting cases, input validation, simulation reproducibility, and agreement between numerical and analytical prices.

Two notebooks present the numerical work:

- **`convergence.ipynb`** studies single-run estimates, repeated-run RMSE convergence, and the effect of antithetic variates across path counts.
- **`results.ipynb`** summarizes the benchmark prices, confidence intervals, price bounds, put–call parity, and repeated-run variance comparison.

## Mathematical framework

### Risk-neutral asset model

Under the Black–Scholes assumptions, the underlying asset follows geometric Brownian motion under the risk-neutral measure:

$$
dS_t = (r-q)S_t\,dt + \sigma S_t\,dW_t,
$$

where $S_t$ is the asset price, $r$ is the continuously compounded risk-free rate, $q$ is the continuous dividend yield, and $\sigma$ is annualized volatility. Over a maturity $T$, the terminal price is lognormally distributed:

$$
S_T = S_0\exp\left((r-q-\tfrac12\sigma^2)T + \sigma\sqrt{T}Z\right),
\qquad Z\sim\mathcal{N}(0,1).
$$

This risk-neutral terminal distribution underlies both the closed-form prices and the simulation experiments.

### Black–Scholes prices

For strike $K$, maturity $T>0$, and volatility $\sigma>0$, define

$$
d_1 = \frac{\ln(S_0/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},
\qquad d_2=d_1-\sigma\sqrt{T},
$$

and let $N(\cdot)$ be the standard normal cumulative distribution function. The European call and put prices are

$$
C = S_0e^{-qT}N(d_1)-Ke^{-rT}N(d_2),
$$

$$
P = Ke^{-rT}N(-d_2)-S_0e^{-qT}N(-d_1).
$$

The implementation also handles expiry by returning intrinsic value and handles zero volatility using the discounted deterministic payoff.

### Conventions and units

Time to maturity is measured in years. Rates, dividend yield, and volatility are annualized decimals: for example, `r=0.05` means 5% per year and `sigma=0.20` means 20% annualized volatility. Rates and dividends are continuously compounded. The dividend yield defaults to zero.

### Risk sensitivities

The Black–Scholes module reports delta, gamma, vega, theta, and rho. Delta and gamma measure sensitivity to spot; vega measures sensitivity to volatility; rho measures sensitivity to the risk-free rate; theta reports annualized calendar-time decay. Vega and rho are derivatives per unit change in their inputs, so a sensitivity to a one-percentage-point change is obtained by multiplying by $0.01$. Greeks are undefined at expiry and at zero volatility, and the module raises an error for those cases.

### Monte Carlo estimator and confidence interval

For a payoff $g(S_T)$, the option value is the discounted risk-neutral expectation

$$
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[g(S_T)].
$$

The Monte Carlo estimator replaces the expectation with a sample average of discounted simulated payoffs. Its standard error is estimated from the sample dispersion, and the reported two-sided confidence interval uses a normal critical value. This interval is an approximation to the estimator's sampling uncertainty, not a guarantee about any individual run.

The simulator accepts a seed or NumPy random generator so experiments can be reproduced. When antithetic sampling is enabled, each standard normal shock $Z$ is paired with $-Z$. The estimator averages the two discounted payoffs within each pair, then estimates its standard error from the independent pair averages. `n_paths` always denotes the total number of terminal price draws and must be even when antithetic sampling is used.

## Numerical findings

For the standard benchmark

$$
S_0=K=100,\quad T=1,\quad r=0.05,\quad \sigma=0.20,\quad q=0,
$$

the analytical prices are approximately **10.4506** for the call and **5.5735** for the put. The results notebook confirms put–call parity and checks the European price bounds:

$$
\max(S_0e^{-qT}-Ke^{-rT},0)\le C\le S_0e^{-qT},
$$

$$
\max(Ke^{-rT}-S_0e^{-qT},0)\le P\le Ke^{-rT}.
$$

The convergence notebook repeats simulations at increasing path counts. In the recorded benchmark experiment, the log–log RMSE slopes were **−0.511 for calls** and **−0.528 for puts**, close to the theoretical $-\tfrac12$ rate. Single-run estimates still fluctuate; the convergence rate describes typical error across repeated simulations.

As an additional numerical cross-check during review, call and put values for a dividend-paying case were compared with independent one-dimensional Gaussian payoff integration. The largest price discrepancy was below **$5\times10^{-10}$**. The five reported Greeks were compared with central finite differences of the price function; the largest discrepancy was below **$8\times10^{-9}$**. The automated suite separately checks Greek benchmark values and pricing relationships.

At 100,000 paths, the saved results show all four ordinary and antithetic call and put confidence intervals containing their analytical benchmark prices. In a separate 100-replication experiment at 20,000 paths, the estimated ordinary-to-antithetic variance ratio was **2.27× for calls** (paired bootstrap 95% interval **1.64–3.19**) and **1.34× for puts** (**0.88–2.01**). The call experiment supports variance reduction; the put interval includes 1, so that experiment is inconclusive about a put variance reduction at the 95% level. These figures are reproducible seeded experiment results, not universal performance guarantees.

## Repository layout

```text
European-Option-Pricing/
├── README.md
├── pyproject.toml
├── src/
│   └── european_option_pricing/
│       ├── __init__.py
│       ├── black_scholes.py
│       └── monte_carlo.py
├── convergence.ipynb
├── results.ipynb
└── tests/
    ├── test_black_scholes.py
    ├── test_monte_carlo.py
    └── test_pricing_properties.py
```

The installable `european_option_pricing` package under `src/` contains the analytical European option pricer, Greeks, terminal-price simulator, and Monte Carlo estimator. The notebooks provide reproducible numerical experiments; the test suite checks the core formulas and estimator behavior.

## Installation

The project requires Python 3.10 or newer. From the project directory (`QF/European-Option-Pricing` in this repository), create and activate a virtual environment, then install the package and its development and notebook dependencies in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

In VS Code, select the `.venv` Python interpreter and, for notebook execution, select the matching `.venv` kernel. Alternatively, start JupyterLab from the project directory:

```bash
jupyter lab
```

## Usage

### Analytical price and Greeks

```python
from european_option_pricing import black_scholes, greeks

call_price = black_scholes(
    S=100.0,
    K=100.0,
    T=1.0,
    r=0.05,
    sigma=0.20,
    option_type="call",
    q=0.0,
)
call_greeks = greeks(
    S=100.0,
    K=100.0,
    T=1.0,
    r=0.05,
    sigma=0.20,
    option_type="call",
    q=0.0,
)

print(f"Call value: {call_price:.4f}")  # 10.4506
print(call_greeks)
```

### Monte Carlo price and confidence interval

```python
from european_option_pricing import monte_carlo_price

estimate = monte_carlo_price(
    S=100.0,
    K=100.0,
    T=1.0,
    r=0.05,
    sigma=0.20,
    option_type="call",
    q=0.0,
    n_paths=100_000,
    confidence_level=0.95,
    rng=42,
    antithetic=True,
)

print(f"Estimate: {estimate.price:.4f}")
print(f"Standard error: {estimate.standard_error:.4f}")
print(f"95% interval: [{estimate.ci_lower:.4f}, {estimate.ci_upper:.4f}]")
```

`monte_carlo_price` returns a `MonteCarloEstimate` with the estimate, standard error, confidence interval, confidence level, and total number of paths. Set `antithetic=False` for ordinary simulation. `simulate_terminal_prices` is also available when an experiment needs terminal asset price draws directly.

### Run the notebooks and tests

Open `convergence.ipynb` and `results.ipynb` in VS Code or JupyterLab and run the cells from top to bottom. With the project virtual environment activated and the package installed, run the automated suite from the project directory:

```bash
python -m pytest -q
```

The tests cover benchmark call and put prices, Greeks, expiry and zero-volatility cases, validation, seeded simulation behavior, antithetic pairing, put–call parity, European price bounds, and numerical agreement between Monte Carlo and analytical prices.

## Assumptions and limitations

- The underlying follows geometric Brownian motion with constant volatility, risk-free rate, and continuous dividend yield.
- Valuation is risk-neutral and assumes frictionless markets, continuous trading, and no transaction costs or funding, liquidity, or market-impact effects.
- The instruments are European options exercisable only at maturity. Early exercise, American options, discrete dividends, stochastic volatility, jumps, and calibration to market data are outside the current scope.
- Monte Carlo estimates contain sampling error. Normal-approximation confidence intervals can be imperfect for finite samples or skewed payoffs.
- Antithetic variates often reduce variance for monotone vanilla option payoffs, but their effectiveness depends on the payoff and model. The reported reduction factors are specific to the seeded experiments.
- Vega and rho are expressed per unit change in volatility and rate, respectively; they are not pre-scaled to percentage points.

## Scope

This project is a focused foundation in option valuation and simulation. It emphasizes correctness, transparent assumptions, and reproducibility before extending to broader models or market-based calibration.
