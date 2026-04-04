"""
Orbit MLP - Surrogate model for galpy orbit integration
Predicts heliocentric XYZ position (parsecs) from initial conditions + time

Input:  (x0, y0, z0, vx0, vy0, vz0, t) - position in parsecs, velocity in pc/Gyr, time in Gyr
Output: (x, y, z)                        - heliocentric position in parsecs
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import json
import time
import os
import glob



# ─── Constants ───────────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 4096
EPOCHS      = 100
LR          = 1e-3
HIDDEN      = [256, 256, 256, 128]   # hidden layer widths
VAL_FRAC    = 0.1                    # 10% validation
TEST_FRAC   = 0.1                    # 10% test
DATA_PATH   = "./data"       # generated training data
NORM_PATH   = "orbit_norm.json"      # saved normalization constants
MODEL_PATH  = "orbit_mlp.pt"        # saved model weights


# ─── Dataset ─────────────────────────────────────────────────────────────────

class OrbitDataset(Dataset):
    """
    Expects a data with shape (N, 10):
        columns 0-6:  x0, y0, z0, vx0, vy0, vz0, t   (inputs)
        columns 7-9:  x, y, z                          (outputs)
    All positions in parsecs, velocities in pc/Gyr, time in Gyr.
    """

    def __init__(self, folder_path: str):
        norm_stats = compute_norm_stats(folder_path)

        files = sorted(glob.glob(folder_path + "/orbit_train_*.npy"))
        data  = np.concatenate([np.load(f, mmap_mode="r") for f in files], axis=0)

        X = data[:, :7].astype(np.float32)
        y = (data[:, 7:] - data[:, :3]).astype(np.float32)

        # normalize inputs
        X = (X - norm_stats["X_mean"]) / norm_stats["X_std"]

        # normalize outputs
        y = (y - norm_stats["y_mean"]) / norm_stats["y_std"]

        print("Inputs normalized, loading into torch.")
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Normalization ────────────────────────────────────────────────────────────

def compute_norm_stats(folder_path: str) -> dict:
    """
    Compute mean and std for inputs and outputs from the full dataset.
    Save to JSON so inference can apply the same transform.
    """
    
    if os.path.exists(NORM_PATH):
        return load_norm_stats(NORM_PATH)
    
    print("Computing norm stats..")

    files = sorted(glob.glob(folder_path + "/orbit_train_*.npy"))

    # Welford's online algorithm
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    n     = 0
    X_mean = np.zeros(7, dtype=np.float64)
    X_M2   = np.zeros(7, dtype=np.float64)
    y_mean = np.zeros(3, dtype=np.float64)
    y_M2   = np.zeros(3, dtype=np.float64)

    for f in files:
        data = np.load(f, mmap_mode="r")
        X    = data[:, :7].astype(np.float64)
        y    = (data[:, 7:] - data[:, :3]).astype(np.float64)

        for row_X, row_y in zip(X, y):
            n += 1
            delta_X  = row_X - X_mean
            X_mean  += delta_X / n
            X_M2    += delta_X * (row_X - X_mean)

            delta_y  = row_y - y_mean
            y_mean  += delta_y / n
            y_M2    += delta_y * (row_y - y_mean)

    X_std = np.maximum(np.sqrt(X_M2 / n), 1e-8)
    y_std = np.maximum(np.sqrt(y_M2 / n), 1e-8)

    stats = {
        "X_mean": X_mean.tolist(),
        "X_std":  X_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std":  y_std.tolist(),
    }

    with open(NORM_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved normalization stats to {NORM_PATH}")
    return stats


def load_norm_stats(filepath: str) -> dict:
    with open(filepath) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


# ─── Model ───────────────────────────────────────────────────────────────────

class OrbitMLP(nn.Module):
    """
    Multi-layer perceptron for orbit prediction.
    7 inputs → hidden layers → 3 outputs.
    """

    def __init__(self, hidden_sizes: list = HIDDEN):
        super().__init__()

        layers = []
        in_size = 7
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Training ────────────────────────────────────────────────────────────────

def train(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, loss_fn) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            predictions = model(X_batch)
            total_loss += loss_fn(predictions, y_batch).item()

    return total_loss / len(loader)


# ─── Loss in parsecs ─────────────────────────────────────────────────────────

def loss_to_parsecs(mse_normalized: float, norm_stats: dict) -> float:
    """
    Convert normalized MSE back to parsecs so loss is interpretable.
    MSE is computed on normalized outputs — undo the std scaling.
    """
    y_std = norm_stats["y_std"]
    # average std across x, y, z dimensions
    avg_std = float(np.mean(y_std))
    # MSE in normalized space = MSE_parsecs / std²
    # so MSE_parsecs = MSE_normalized * std²
    rmse_parsecs = np.sqrt(mse_normalized) * avg_std
    return rmse_parsecs


# ─── Inference ───────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH, norm_path: str = NORM_PATH):
    """Load trained model and normalization stats for inference."""
    norm_stats = load_norm_stats(norm_path)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = OrbitMLP(hidden_sizes=checkpoint["hidden_sizes"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, norm_stats


def predict(model, norm_stats: dict, x0, y0, z0, vx0, vy0, vz0, t) -> tuple:
    """
    Predict heliocentric XYZ position in parsecs.

    Args:
        x0, y0, z0:     initial galactocentric position (parsecs)
        vx0, vy0, vz0:  initial galactocentric velocity (pc/Gyr)
        t:              target time (Gyr)

    Returns:
        (x, y, z) heliocentric position in parsecs — numpy arrays
    """
    X = np.array([[x0, y0, z0, vx0, vy0, vz0, t]], dtype=np.float32)
    X = (X - norm_stats["X_mean"]) / norm_stats["X_std"]
    X_tensor = torch.from_numpy(X).to(DEVICE)

    with torch.no_grad():
        y_norm = model(X_tensor).cpu().numpy()

    y = y_norm * norm_stats["y_std"] + norm_stats["y_mean"]
    return y[0, 0], y[0, 1], y[0, 2]


def predict_batch(model, norm_stats: dict, inputs: np.ndarray) -> np.ndarray:
    """
    Batch inference for N stars.

    Args:
        inputs: (N, 7) array of [x0, y0, z0, vx0, vy0, vz0, t]

    Returns:
        (N, 3) array of [x, y, z] in parsecs
    """
    X = inputs.astype(np.float32)
    X = (X - norm_stats["X_mean"]) / norm_stats["X_std"]
    X_tensor = torch.from_numpy(X).to(DEVICE)

    with torch.no_grad():
        y_norm = model(X_tensor).cpu().numpy()

    return y_norm * norm_stats["y_std"] + norm_stats["y_mean"]



# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    torch.set_float32_matmul_precision("high")

    # --- dataset ---
    print("Loading dataset...")
    full_dataset = OrbitDataset(DATA_PATH)
    n = len(full_dataset)
    return
    n_val  = int(n * VAL_FRAC)
    n_test = int(n * TEST_FRAC)
    n_train = n - n_val - n_test

    # TODO: Input 3 sets instead
    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- model ---
    model = OrbitMLP(hidden_sizes=HIDDEN).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    # learning rate scheduler — halves LR if val loss stops improving for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- training loop ---
    best_val_loss = float("inf")
    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn)
        val_loss   = evaluate(model, val_loader, loss_fn)

        scheduler.step(val_loss)

        train_pc = loss_to_parsecs(train_loss, norm_stats)
        val_pc   = loss_to_parsecs(val_loss,   norm_stats)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.6f} ({train_pc:.2f} pc)  "
              f"val_loss={val_loss:.6f} ({val_pc:.2f} pc)  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"time={elapsed:.1f}s")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "hidden_sizes": HIDDEN,
                "norm_path": NORM_PATH,
            }, MODEL_PATH)

    # --- final test evaluation ---
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    test_loss = evaluate(model, test_loader, loss_fn)
    test_pc   = loss_to_parsecs(test_loss, norm_stats)
    print(f"\nTest loss: {test_loss:.6f}  ({test_pc:.4f} parsecs RMSE)")
    print(f"Target:    0.25 pc² MSE  (0.50 pc RMSE)")
    print(f"{'✓ Target hit' if test_pc <= 0.5 else '✗ Target not hit — consider more data or larger network'}")


if __name__ == "__main__":
    main()
