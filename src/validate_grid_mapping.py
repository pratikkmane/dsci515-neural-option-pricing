
"""Validate the strike-maturity ordering of the Heston dataset."""

import gzip
from pathlib import Path

import numpy as np

from heston import (
    heston_call,
    implied_volatility,
    black_scholes_call,
)



ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "HestonTrainSet.txt.gz"

STRIKES = np.arange(0.5, 1.51, 0.1)
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])

SPOT = 1.0
RATE = 0.0


def load_dataset(path: Path) -> np.ndarray:
    """Load the dataset stored as a gzipped NumPy array."""
    with gzip.open(path, "rb") as file:
        return np.load(file)


def calculate_reference_grid(parameters):
    """Calculate Heston implied volatility for each strike/maturity."""

    xi0, rho, sigma, theta, kappa = parameters

    reference = np.empty((len(MATURITIES), len(STRIKES)))

    for i, maturity in enumerate(MATURITIES):
        for j, strike in enumerate(STRIKES):
            price = heston_call(
                spot=SPOT,
                strike=float(strike),
                maturity=float(maturity),
                xi0=float(xi0),
                rho=float(rho),
                sigma=float(sigma),
                theta=float(theta),
                kappa=float(kappa),
                rate=RATE,
            )

            reference[i, j] = implied_volatility(
                price=price,
                spot=SPOT,
                strike=float(strike),
                maturity=float(maturity),
                rate=RATE,
            )

    return reference


def main():
    dataset = load_dataset(DATA_PATH)
    parameters = dataset[0, :5]
    observed = dataset[0, 5:]

    print("First dataset parameters:", parameters)
    print("Calculating 88 reference implied volatilities...")

    reference = calculate_reference_grid(parameters)
    
    print("\nReference price versus dataset-implied price:")

    for maturity, strike in [
        (0.1, 0.6),
        (0.1, 1.0),
        (0.1, 1.5),
        (0.3, 0.5),
    ]:
        i = np.where(np.isclose(MATURITIES, maturity))[0][0]
        j = np.where(np.isclose(STRIKES, strike))[0][0]

        dataset_iv = observed[i * len(STRIKES) + j]

        dataset_price = black_scholes_call(
            SPOT, strike, maturity, dataset_iv, RATE
        )

        reference_price = heston_call(
            SPOT, strike, maturity,
            *parameters, rate=RATE
        )

        print(
            f"T={maturity:.1f}, K={strike:.1f}, "
            f"dataset_price={dataset_price:.10f}, "
            f"reference_price={reference_price:.10f}, "
            f"price_difference={reference_price - dataset_price:+.10f}"
        )

    
    print("\nHeston call-price bounds diagnostic:")

    xi0, rho, sigma, theta, kappa = parameters

    for maturity, strike in [
        (0.1, 0.5),
        (0.1, 0.6),
        (0.1, 1.0),
        (0.1, 1.5),
        (0.3, 0.5),
    ]:
        price = heston_call(
            spot=SPOT,
            strike=strike,
            maturity=maturity,
            xi0=xi0,
            rho=rho,
            sigma=sigma,
            theta=theta,
            kappa=kappa,
            rate=RATE,
        )

        lower = max(
            SPOT - strike * np.exp(-RATE * maturity), 0.0
        )
        upper = SPOT

        print(
            f"T={maturity:.1f}, K={strike:.1f}, "
            f"price={price:.10f}, "
            f"lower={lower:.10f}, "
            f"upper={upper:.10f}, "
            f"within_bounds={lower <= price <= upper}"
        )

    observed_grid = observed.reshape(
        len(MATURITIES), len(STRIKES)
    )

    absolute_errors = np.abs(reference - observed_grid)

    valid = np.isfinite(absolute_errors)
    ranked_indices = np.argsort(
        np.where(valid, absolute_errors, -np.inf).ravel()
    )[::-1]

    print("\nTop 10 reference pricing discrepancies:")

    for flat_index in ranked_indices[:10]:
        i, j = np.unravel_index(
            flat_index, absolute_errors.shape
        )

        print(
            f"T={MATURITIES[i]:.1f}, "
            f"K={STRIKES[j]:.1f}, "
            f"dataset={observed_grid[i, j]:.6f}, "
            f"reference={reference[i, j]:.6f}, "
            f"error={absolute_errors[i, j]:.6f}"
        )

    
    print("\nAt-the-money comparison:")
    for i, maturity in enumerate(MATURITIES):
        j = 5  # Strike = 1.0
        index = i * len(STRIKES) + j

        print(
            f"T={maturity:.1f}, "
            f"dataset={observed[index]:.6f}, "
            f"reference={reference[i, j]:.6f}, "
            f"difference={reference[i, j] - observed[index]:+.6f}"
        )

    
    failed_points = np.argwhere(~np.isfinite(reference))

    for i, j in failed_points:
        print(
            f"Failed point: maturity={MATURITIES[i]}, "
            f"strike={STRIKES[j]}"
        )

    print("Finite reference values:", np.isfinite(reference).sum())


    # Candidate A: maturities are outer groups, strikes vary fastest.
    maturity_first = reference.ravel(order="C")

    # Candidate B: strikes are outer groups, maturities vary fastest.
    strike_first = reference.ravel(order="F")

    
    for name, candidate in [
        ("Maturity-first", maturity_first),
        ("Strike-first", strike_first),
    ]:
        valid = np.isfinite(candidate) & np.isfinite(observed)

        errors = np.abs(candidate[valid] - observed[valid])

        print(f"\n{name}:")
        print(f"Valid comparison points: {valid.sum()} / 88")

        if errors.size == 0:
            print("No valid points available.")
            continue

        print(f"Mean absolute IV error: {np.mean(errors):.8f}")
        print(f"Maximum absolute IV error: {np.max(errors):.8f}")

        

    

    print("\nDataset output index 10:", observed[10])
    print("Maturity-first reference index 10:", maturity_first[10])
    print("Strike-first reference index 10:", strike_first[10])


if __name__ == "__main__":
    main()
