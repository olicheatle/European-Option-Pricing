import math

import pytest

from european_option_pricing.black_scholes import black_scholes
from european_option_pricing.monte_carlo import monte_carlo_price


@pytest.mark.parametrize(
    ("S", "K", "T", "r", "q", "sigma"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.20),
        (125.0, 110.0, 0.75, 0.03, 0.02, 0.35),
        (80.0, 95.0, 2.0, 0.01, 0.04, 0.15),
    ],
)
def test_black_scholes_put_call_parity(S, K, T, r, q, sigma):
    call = black_scholes(S, K, T, r, sigma, "call", q=q)
    put = black_scholes(S, K, T, r, sigma, "put", q=q)
    discounted_forward = S * math.exp(-q * T) - K * math.exp(-r * T)

    assert call - put == pytest.approx(discounted_forward, abs=1e-10)


@pytest.mark.parametrize(
    ("S", "K", "T", "r", "q", "sigma"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.20),
        (125.0, 110.0, 0.75, 0.03, 0.02, 0.35),
        (80.0, 95.0, 2.0, 0.01, 0.04, 0.15),
    ],
)
def test_black_scholes_prices_respect_european_no_arbitrage_bounds(
    S, K, T, r, q, sigma
):
    discounted_spot = S * math.exp(-q * T)
    discounted_strike = K * math.exp(-r * T)
    call = black_scholes(S, K, T, r, sigma, "call", q=q)
    put = black_scholes(S, K, T, r, sigma, "put", q=q)

    assert max(discounted_spot - discounted_strike, 0.0) <= call
    assert call <= discounted_spot
    assert max(discounted_strike - discounted_spot, 0.0) <= put
    assert put <= discounted_strike


@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_estimate_agrees_with_analytical_price(
    option_type, antithetic
):
    params = {"S": 100.0, "K": 105.0, "T": 1.25, "r": 0.04, "sigma": 0.25, "q": 0.015}
    analytic = black_scholes(**params, option_type=option_type)
    estimate = monte_carlo_price(
        **params,
        option_type=option_type,
        n_paths=100_000,
        confidence_level=0.99,
        rng=94721,
        antithetic=antithetic,
    )

    # A four-standard-error tolerance is robust to Monte Carlo sampling noise.
    assert abs(estimate.price - analytic) <= 4 * estimate.standard_error
    assert estimate.ci_lower <= analytic <= estimate.ci_upper


@pytest.mark.parametrize("antithetic", [False, True])
def test_monte_carlo_put_call_parity(antithetic):
    params = {"S": 100.0, "K": 105.0, "T": 1.25, "r": 0.04, "sigma": 0.25, "q": 0.015}
    call = monte_carlo_price(
        **params, option_type="call", n_paths=100_000, rng=125, antithetic=antithetic
    )
    put = monte_carlo_price(
        **params, option_type="put", n_paths=100_000, rng=125, antithetic=antithetic
    )
    parity = params["S"] * math.exp(-params["q"] * params["T"])
    parity -= params["K"] * math.exp(-params["r"] * params["T"])
    tolerance = 4 * math.hypot(call.standard_error, put.standard_error)

    assert abs((call.price - put.price) - parity) <= tolerance

