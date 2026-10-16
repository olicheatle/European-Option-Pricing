import math

import pytest

from european_option_pricing.black_scholes import black_scholes, greeks


BENCHMARK = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20}


def test_benchmark_call_and_put_prices():
    assert black_scholes(**BENCHMARK, option_type="call") == pytest.approx(10.4506, abs=1e-4)
    assert black_scholes(**BENCHMARK, option_type="put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity_with_dividend_yield():
    q = 0.025
    call = black_scholes(**BENCHMARK, option_type="call", q=q)
    put = black_scholes(**BENCHMARK, option_type="put", q=q)
    parity_value = BENCHMARK["S"] * math.exp(-q * BENCHMARK["T"]) - BENCHMARK["K"] * math.exp(
        -BENCHMARK["r"] * BENCHMARK["T"]
    )
    assert call - put == pytest.approx(parity_value)


@pytest.mark.parametrize(
    ("option_type", "expected"),
    [("call", 15.0), ("put", 0.0)],
)
def test_price_at_expiry(option_type, expected):
    assert black_scholes(105, 90, 0, 0.03, 0.2, option_type) == expected


@pytest.mark.parametrize(
    ("option_type", "expected"),
    [
        ("call", max(100 * math.exp(-0.02) - 95 * math.exp(-0.04), 0.0)),
        ("put", max(95 * math.exp(-0.04) - 100 * math.exp(-0.02), 0.0)),
    ],
)
def test_zero_volatility_discounted_deterministic_payoff(option_type, expected):
    assert black_scholes(100, 95, 1, 0.04, 0.0, option_type, q=0.02) == pytest.approx(expected)


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
    ],
)
def test_invalid_inputs_raise_value_error(kwargs):
    inputs = {**BENCHMARK, "option_type": "call", **kwargs}
    with pytest.raises(ValueError):
        black_scholes(**inputs)


def test_greeks_benchmark_values():
    call = greeks(**BENCHMARK, option_type="call")
    put = greeks(**BENCHMARK, option_type="put")

    assert call["delta"] == pytest.approx(0.63683065)
    assert put["delta"] == pytest.approx(-0.36316935)
    assert call["gamma"] == pytest.approx(0.01876202)
    assert put["gamma"] == pytest.approx(call["gamma"])
    assert call["vega"] == pytest.approx(37.52403469)
    assert put["vega"] == pytest.approx(call["vega"])
    assert call["theta"] == pytest.approx(-6.41402755)
    assert put["theta"] == pytest.approx(-1.65788042)
    assert call["rho"] == pytest.approx(53.23248155)
    assert put["rho"] == pytest.approx(-41.89046090)


@pytest.mark.parametrize("kwargs", [{"T": 0.0}, {"sigma": 0.0}, {"option_type": "other"}])
def test_greeks_reject_undefined_or_invalid_inputs(kwargs):
    inputs = {**BENCHMARK, "option_type": "call", **kwargs}
    with pytest.raises(ValueError):
        greeks(**inputs)
