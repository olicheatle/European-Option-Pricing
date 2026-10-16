"""Black–Scholes pricing and risk sensitivities for European options.

Rates, volatility, and dividend yield are continuously compounded/annualised;
time is measured in years. Greeks are derivatives with respect to the named
input. Vega and rho are per unit change (for example, 0.01 means one percentage
point); theta is the calendar-time decay per year.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _validate_inputs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float,
) -> OptionType:
    """Validate shared model inputs and return a normalized option type."""
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
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    return option_type  # type: ignore[return-value]


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """Return the Black–Scholes price of a European call or put.

    Args:
        S: Current underlying asset price; must be positive.
        K: Strike price; must be positive.
        T: Time to maturity in years; must be non-negative.
        r: Continuously compounded annual risk-free rate.
        sigma: Annualised volatility; must be non-negative.
        option_type: ``"call"`` or ``"put"``.
        q: Continuously compounded annual dividend yield.

    At expiry, returns intrinsic value. At zero volatility, returns the
    discounted payoff under the deterministic risk-neutral asset path.
    """
    option_type = _validate_inputs(S, K, T, r, sigma, option_type, q)

    if T == 0:
        return float(max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0))

    discounted_spot = S * math.exp(-q * T)
    discounted_strike = K * math.exp(-r * T)
    if sigma == 0:
        if option_type == "call":
            return float(max(discounted_spot - discounted_strike, 0.0))
        return float(max(discounted_strike - discounted_spot, 0.0))

    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option_type == "call":
        price = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    else:
        price = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
    return float(price)


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> dict[str, float]:
    """Return delta, gamma, vega, theta, and rho for a European option.

    Delta and gamma are derivatives with respect to spot. Vega and rho are
    derivatives with respect to volatility and the risk-free rate, respectively
    (per 1.00 change, not per percentage point). Theta is the derivative with
    respect to calendar time, so it is typically negative for long options.

    Greeks are undefined at expiry or when volatility is zero; in either case
    this function raises ``ValueError``. Pricing itself supports both cases.
    """
    option_type = _validate_inputs(S, K, T, r, sigma, option_type, q)
    if T == 0:
        raise ValueError("Greeks are undefined at expiry (T must be positive).")
    if sigma == 0:
        raise ValueError("Greeks are undefined at zero volatility (sigma must be positive).")

    sqrt_t = math.sqrt(T)
    vol_sqrt_t = sigma * sqrt_t
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    discount_q = math.exp(-q * T)
    discount_r = math.exp(-r * T)
    density = norm.pdf(d1)

    gamma = discount_q * density / (S * vol_sqrt_t)
    vega = S * discount_q * density * sqrt_t

    if option_type == "call":
        delta = discount_q * norm.cdf(d1)
        theta = (
            -S * discount_q * density * sigma / (2 * sqrt_t)
            + q * S * discount_q * norm.cdf(d1)
            - r * K * discount_r * norm.cdf(d2)
        )
        rho = K * T * discount_r * norm.cdf(d2)
    else:
        delta = discount_q * (norm.cdf(d1) - 1)
        theta = (
            -S * discount_q * density * sigma / (2 * sqrt_t)
            - q * S * discount_q * norm.cdf(-d1)
            + r * K * discount_r * norm.cdf(-d2)
        )
        rho = -K * T * discount_r * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


if __name__ == "__main__":
    benchmark = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20}
    for kind in ("call", "put"):
        price = black_scholes(**benchmark, option_type=kind)
        print(f"{kind.capitalize()} price: {price:.4f}")
        print(f"Greeks: {greeks(**benchmark, option_type=kind)}")
