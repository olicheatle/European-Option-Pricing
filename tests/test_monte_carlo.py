import numpy as np
import pytest

from european_option_pricing.black_scholes import black_scholes
from european_option_pricing.monte_carlo import monte_carlo_price, simulate_terminal_prices


PARAMS = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}


def test_terminal_price_simulation_is_reproducible_and_has_expected_shape():
    first = simulate_terminal_prices(100, 1, 0.05, 0.2, 2_000, q=0.01, rng=123)
    second = simulate_terminal_prices(100, 1, 0.05, 0.2, 2_000, q=0.01, rng=123)

    assert first.shape == (2_000,)
    assert np.array_equal(first, second)
    assert np.all(first > 0)


def test_terminal_price_sample_mean_matches_risk_neutral_forward():
    q = 0.02
    prices = simulate_terminal_prices(100, 1.5, 0.04, 0.25, 200_000, q=q, rng=321)
    expected_mean = 100 * np.exp((0.04 - q) * 1.5)
    assert prices.mean() == pytest.approx(expected_mean, rel=0.003)


def test_antithetic_terminal_prices_are_reproducible_and_paired():
    params = (100, 1.0, 0.05, 0.2, 2_000)
    first = simulate_terminal_prices(*params, q=0.01, rng=123, antithetic=True)
    second = simulate_terminal_prices(*params, q=0.01, rng=123, antithetic=True)
    expected_pair_product = 100**2 * np.exp(2 * (0.05 - 0.01 - 0.5 * 0.2**2))

    assert np.array_equal(first, second)
    assert first.shape == (2_000,)
    assert first[::2] * first[1::2] == pytest.approx(expected_pair_product)


def test_antithetic_sampling_requires_even_path_count():
    with pytest.raises(ValueError, match="even"):
        simulate_terminal_prices(100, 1, 0.05, 0.2, 101, antithetic=True)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_estimate_is_reproducible_and_contains_black_scholes_price(option_type):
    result = monte_carlo_price(**PARAMS, option_type=option_type, n_paths=200_000, rng=2026)
    analytic = black_scholes(**PARAMS, option_type=option_type)

    repeated = monte_carlo_price(**PARAMS, option_type=option_type, n_paths=200_000, rng=2026)
    assert result == repeated
    assert result.ci_lower <= analytic <= result.ci_upper
    assert result.standard_error > 0
    assert result.n_paths == 200_000


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_antithetic_price_is_reproducible_and_has_correct_path_count(option_type):
    result = monte_carlo_price(
        **PARAMS, option_type=option_type, n_paths=20_000, rng=2026, antithetic=True
    )
    repeated = monte_carlo_price(
        **PARAMS, option_type=option_type, n_paths=20_000, rng=2026, antithetic=True
    )
    analytic = black_scholes(**PARAMS, option_type=option_type)

    assert result == repeated
    assert result.ci_lower <= analytic <= result.ci_upper
    assert result.n_paths == 20_000
    assert result.standard_error > 0


def test_antithetic_variates_reduce_call_estimator_variance():
    ordinary_prices = []
    antithetic_prices = []
    for seed in range(30):
        ordinary_prices.append(
            monte_carlo_price(
                **PARAMS, option_type="call", n_paths=10_000, rng=seed
            ).price
        )
        antithetic_prices.append(
            monte_carlo_price(
                **PARAMS, option_type="call", n_paths=10_000, rng=seed, antithetic=True
            ).price
        )

    assert np.var(antithetic_prices, ddof=1) < np.var(ordinary_prices, ddof=1)


@pytest.mark.parametrize(
    ("option_type", "expected"),
    [("call", 15.0), ("put", 0.0)],
)
def test_expiry_returns_intrinsic_value_with_zero_width_interval(option_type, expected):
    result = monte_carlo_price(105, 90, 0, 0.03, 0.2, option_type, n_paths=10, rng=1)

    assert result.price == pytest.approx(expected)
    assert result.standard_error == 0.0
    assert result.ci_lower == result.ci_upper == result.price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_zero_volatility_price_matches_black_scholes(option_type):
    params = {"S": 100, "K": 95, "T": 1, "r": 0.04, "sigma": 0.0, "q": 0.02}
    result = monte_carlo_price(**params, option_type=option_type, n_paths=10, rng=1)
    analytic = black_scholes(**params, option_type=option_type)

    assert result.price == pytest.approx(analytic)
    assert result.standard_error == 0.0
    assert result.ci_lower == result.ci_upper == result.price


@pytest.mark.parametrize(
    "kwargs",
    [
        {"S": 0},
        {"K": -1},
        {"T": -0.1},
        {"sigma": -0.1},
        {"r": float("nan")},
        {"q": float("inf")},
        {"option_type": "straddle"},
        {"n_paths": 0},
        {"n_paths": 1},
        {"n_paths": 10.5},
        {"confidence_level": 1.0},
    ],
)
def test_invalid_inputs_raise(kwargs):
    inputs = {**PARAMS, "option_type": "call", "n_paths": 100, **kwargs}
    with pytest.raises((TypeError, ValueError)):
        monte_carlo_price(**inputs)
