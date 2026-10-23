
"""Inspect the saved Heston reproduction dataset."""

import argparse
import gzip
import inspect
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from reproduce_heston import LB, UB

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "HestonTrainSet.txt.gz"
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "results" / "reproduction_weights.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the Heston reproduction dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the gzipped dataset file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to trained model weights.",
    )
    return parser.parse_args()


args = parse_args()
data_path = args.data.resolve()
weights_path = args.weights.resolve()

if not data_path.exists():
    raise FileNotFoundError(f"Dataset file not found: {data_path}")
if not weights_path.exists():
    raise FileNotFoundError(f"Weights file not found: {weights_path}")

with gzip.open(data_path, "rb") as file:
    data = np.load(file)

parameters = data[:, :5]
implied_vols = data[:, 5:]

print("Dataset shape:", data.shape)
print("Parameters shape:", parameters.shape)
print("Implied volatility shape:", implied_vols.shape)

print("\nParameter ranges:")
names = ["xi0", "rho", "sigma", "theta", "kappa"]

for i, name in enumerate(names):
    print(
        f"{name}: "
        f"min={parameters[:, i].min():.6f}, "
        f"max={parameters[:, i].max():.6f}"
    )

print("\nImplied volatility statistics:")
print("Minimum:", implied_vols.min())
print("Maximum:", implied_vols.max())
print("Mean:", implied_vols.mean())




# Reproduce the original train/test split
P_train, P_test, S_train, S_test = train_test_split(
    parameters,
    implied_vols,
    test_size=0.15,
    random_state=42
)

# Scale reference implied volatility surfaces
scaler = StandardScaler().fit(S_train)

# Scale Heston parameters to [-1, 1]
X_test = 2 * (P_test - (UB + LB) / 2) / (UB - LB)

# Rebuild the original neural network
torch.set_default_dtype(torch.float64)

model = torch.nn.Sequential(
    torch.nn.Linear(5, 30),
    torch.nn.ELU(),
    torch.nn.Linear(30, 30),
    torch.nn.ELU(),
    torch.nn.Linear(30, 30),
    torch.nn.ELU(),
    torch.nn.Linear(30, 88)
)

# Load previously trained weights
model.load_state_dict(
    torch.load(
        weights_path,
        **(
            {"map_location": "cpu", "weights_only": True}
            if "weights_only" in inspect.signature(torch.load).parameters
            else {"map_location": "cpu"}
        ),
    )
)
model.eval()

# Predict implied volatilities
with torch.no_grad():
    scaled_predictions = model(
        torch.tensor(X_test, dtype=torch.float64)
    ).numpy()

predictions = scaler.inverse_transform(scaled_predictions)

# Calculate relative prediction errors
relative_errors = (
    100 * np.abs(S_test - predictions) / S_test
)

# Find the largest error
surface_idx, grid_idx = np.unravel_index(
    np.argmax(relative_errors),
    relative_errors.shape
)


# Compare errors across initial-variance groups
xi0_test = P_test[:, 0]

# Maximum relative error across each 88-point surface
surface_max_errors = relative_errors.max(axis=1)

# Divide test surfaces into four xi0 groups
quartile_edges = np.quantile(
    xi0_test, [0, 0.25, 0.50, 0.75, 1.0]
)

print("\nPrediction errors by xi0 quartile:")

for i in range(4):
    lower = quartile_edges[i]
    upper = quartile_edges[i + 1]

    if i < 3:
        mask = (xi0_test >= lower) & (xi0_test < upper)
    else:
        mask = (xi0_test >= lower) & (xi0_test <= upper)

    print(
        f"Quartile {i + 1}: "
        f"xi0=[{lower:.6f}, {upper:.6f}], "
        f"mean max error={surface_max_errors[mask].mean():.4f}%, "
        f"worst error={surface_max_errors[mask].max():.4f}%"
    )


print("\nLargest prediction error:")
print("Surface index:", surface_idx)
print("Grid index:", grid_idx)
print("Relative error (%):", relative_errors[surface_idx, grid_idx])
print("True implied volatility:", S_test[surface_idx, grid_idx])
print("Predicted implied volatility:", predictions[surface_idx, grid_idx])
print("Heston parameters:", P_test[surface_idx])
