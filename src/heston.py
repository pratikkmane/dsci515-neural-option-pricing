
"""Reference Heston European call pricing using numerical integration."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm


def black_scholes_call(spot, strike, maturity, volatility, rate=0.0):
    """Black-Scholes European call price, without dividends."""
    if maturity <= 0:
        return max(spot - strike, 0.0)

    if volatility <= 0:
        return max(spot - strike * np.exp(-rate * maturity), 0.0)

    d1 = (
        np.log(spot / strike)
        + (rate + 0.5 * volatility**2) * maturity
    ) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)

    return spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)


def heston_characteristic_function(
    u, maturity, xi0, rho, sigma, theta, kappa, rate=0.0
):
    """Characteristic function of the Heston log return."""
    iu = 1j * u

    d = np.sqrt(
        (kappa - rho * sigma * iu)**2
        + sigma**2 * (u**2 + iu)
    )

    g = (kappa - rho * sigma * iu - d) / (
        kappa - rho * sigma * iu + d
    )

    exp_dt = np.exp(-d * maturity)

    c = (
        rate * iu * maturity
        + (kappa * theta / sigma**2)
        * (
            (kappa - rho * sigma * iu - d) * maturity
            - 2 * np.log((1 - g * exp_dt) / (1 - g))
        )
    )

    d_term = (
        (kappa - rho * sigma * iu - d) / sigma**2
        * ((1 - exp_dt) / (1 - g * exp_dt))
    )

    return np.exp(c + d_term * xi0)


def heston_call(
    spot, strike, maturity, xi0, rho, sigma, theta, kappa, rate=0.0
):
    """Price a European call using the Heston characteristic function."""
    if maturity <= 0:
        return max(spot - strike, 0.0)

    log_moneyness = np.log(spot / strike)
    phi_minus_i = heston_characteristic_function(
        -1j, maturity, xi0, rho, sigma, theta, kappa, rate
    )

    def p1_integrand(u):
        phi = heston_characteristic_function(
            u - 1j, maturity, xi0, rho, sigma, theta, kappa, rate
        )
        value = np.exp(1j * u * log_moneyness) * phi / (
            1j * u * phi_minus_i
        )
        return np.real(value)

    def p2_integrand(u):
        phi = heston_characteristic_function(
            u, maturity, xi0, rho, sigma, theta, kappa, rate
        )
        value = np.exp(1j * u * log_moneyness) * phi / (1j * u)
        return np.real(value)

    
    p1 = 0.5 + quad(
        p1_integrand,
        1e-8,
        500,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=500,
    )[0] / np.pi

    p2 = 0.5 + quad(
        p2_integrand,
        1e-8,
        500,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=500,
    )[0] / np.pi


    return spot * p1 - strike * np.exp(-rate * maturity) * p2


def implied_volatility(
    price, spot, strike, maturity, rate=0.0
):
    """Invert Black-Scholes to obtain implied volatility."""
    lower_bound = max(
        spot - strike * np.exp(-rate * maturity), 0.0
    )
    upper_bound = spot

    if not lower_bound < price < upper_bound:
        return np.nan

    def objective(volatility):
        return (
            black_scholes_call(
                spot, strike, maturity, volatility, rate
            )
            - price
        )

    try:
        return brentq(objective, 1e-6, 5.0)
    except ValueError:
        return np.nan

if __name__ == "__main__":
    price = heston_call(
        spot=1.0,
        strike=1.0,
        maturity=1.0,
        xi0=0.04,
        rho=-0.7,
        sigma=0.3,
        theta=0.04,
        kappa=2.0,
    )

    iv = implied_volatility(
        price, spot=1.0, strike=1.0, maturity=1.0
    )

    print(f"Heston call price: {price:.8f}")
    print(f"Implied volatility: {iv:.8f}")

    print("\nBlack-Scholes limiting-case test:")

    heston_price = heston_call(
        spot=1.0,
        strike=1.0,
        maturity=1.0,
        xi0=0.04,
        rho=-0.7,
        sigma=0.001,
        theta=0.04,
        kappa=2.0,
    )

    bs_price = black_scholes_call(
        spot=1.0,
        strike=1.0,
        maturity=1.0,
        volatility=0.2,
    )
    
    print("\nCharacteristic-function consistency test:")

    phi_zero = heston_characteristic_function(
        0.0, 1.0, 0.04, -0.7, 0.3, 0.04, 2.0
    )

    phi_minus_i = heston_characteristic_function(
        -1j, 1.0, 0.04, -0.7, 0.3, 0.04, 2.0
    )

    print(f"phi(0): {phi_zero}")
    print(f"phi(-i): {phi_minus_i}")
    print(f"Error at zero: {abs(phi_zero - 1):.12f}")
    print(f"Error at -i: {abs(phi_minus_i - 1):.12f}")


    print(f"Heston price: {heston_price:.8f}")
    print(f"Black-Scholes price: {bs_price:.8f}")
    print(f"Absolute difference: {abs(heston_price - bs_price):.8f}")
