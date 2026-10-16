"""Monte Carlo simulation and pricing for European options under Black–Scholes.

The simulator samples risk-neutral terminal asset prices directly. The pricer
discounts each terminal payoff and reports a normal-approximation confidence
interval for the estimated present value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MonteCarloEstimate:
    """Monte Carlo price estimate and two-sided confidence interval."""

    price: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_paths: int


def _validate_market_inputs(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> None:
    values = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q}
    for name, value in values.items():
        try:
            finite = math.isfinite(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a real number.") from exc
        if not finite:
            raise ValueError(f"{name} must be finite.")
    if S <= 0:
        raise ValueError("S must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T < 0:
        raise ValueError("T cannot be negative.")
    if sigma < 0:
        raise ValueError("Volatility cannot be negative.")


def _validate_option_type(option_type: str) -> OptionType:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    return option_type  # type: ignore[return-value]


def _validate_path_count(n_paths: int) -> None:
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)):
        raise TypeError("n_paths must be an integer.")
    if n_paths < 1:
        raise ValueError("n_paths must be positive.")


def simulate_terminal_prices(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    q: float = 0.0,
    rng: np.random.Generator | int | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate risk-neutral terminal prices for a dividend-paying asset.

    Args:
        S: Current asset price; must be positive.
        T: Time to maturity in years; must be non-negative.
        r: Continuously compounded annual risk-free rate.
        sigma: Annualised volatility; must be non-negative.
        n_paths: Total number of terminal price draws.
        q: Continuously compounded annual dividend yield.
        rng: NumPy generator, seed, or ``None`` for a fresh random stream.
        antithetic: If true, use adjacent ``Z`` and ``-Z`` normal draws.
            This requires an even ``n_paths``.

    Returns:
        A one-dimensional NumPy array of length ``n_paths``.
    """
    _validate_market_inputs(S, 1.0, T, r, sigma, q)
    _validate_path_count(n_paths)
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even when antithetic variates are enabled.")

    generator = np.random.default_rng(rng)
    if T == 0 or sigma == 0:
        terminal = S * math.exp((r - q - 0.5 * sigma**2) * T)
        return np.full(n_paths, terminal, dtype=float)

    if antithetic:
        z_half = generator.standard_normal(n_paths // 2)
        z = np.column_stack((z_half, -z_half)).ravel()
    else:
        z = generator.standard_normal(n_paths)
    exponent = (r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z
    return S * np.exp(exponent)


def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    n_paths: int = 100_000,
    confidence_level: float = 0.95,
    rng: np.random.Generator | int | None = None,
    antithetic: bool = False,
) -> MonteCarloEstimate:
    """Estimate a European option value and its confidence interval.

    By default, the estimator is the sample mean of discounted risk-neutral
    payoffs. With ``antithetic=True``, adjacent paths use normal shocks ``Z``
    and ``-Z``; each pair's average payoff is one independent observation for
    estimating the standard error. In both cases, standard error uses the
    sample standard deviation (``ddof=1``) of the independent observations.
    The two-sided interval uses a normal critical value and is therefore an
    asymptotic confidence interval. Pass a seed or generator for reproducible
    simulations. Antithetic sampling requires an even ``n_paths``.

    At expiry and at zero volatility, all paths have the same payoff, giving
    an exact price and a zero-width interval.
    """
    _validate_market_inputs(S, K, T, r, sigma, q)
    option_type = _validate_option_type(option_type)
    _validate_path_count(n_paths)
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2 to estimate a confidence interval.")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be finite and strictly between 0 and 1.")

    terminal_prices = simulate_terminal_prices(
        S, T, r, sigma, n_paths, q=q, rng=rng, antithetic=antithetic
    )
    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0.0)

    discounted_payoffs = math.exp(-r * T) * payoffs
    if antithetic:
        independent_observations = discounted_payoffs.reshape(-1, 2).mean(axis=1)
    else:
        independent_observations = discounted_payoffs
    price = float(np.mean(independent_observations))
    standard_error = float(
        np.std(independent_observations, ddof=1) / math.sqrt(independent_observations.size)
    )
    critical_value = float(norm.ppf(0.5 + confidence_level / 2))
    half_width = critical_value * standard_error
    return MonteCarloEstimate(
        price=price,
        standard_error=standard_error,
        ci_lower=price - half_width,
        ci_upper=price + half_width,
        confidence_level=float(confidence_level),
        n_paths=int(n_paths),
    )


if __name__ == "__main__":
    result = monte_carlo_price(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.20,
        option_type="call",
        n_paths=100_000,
        rng=42,
    )
    print(f"Monte Carlo call price: {result.price:.4f}")
    print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
