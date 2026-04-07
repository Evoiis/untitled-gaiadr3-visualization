"""
Orbit MLP - Surrogate model for galpy orbit integration
Predicts heliocentric XYZ position (parsecs) from initial conditions + time

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.amp import autocast, GradScaler
import json
import time
import math
import os
import glob

from collections import Counter



# ─── Constants ───────────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 100000

# TODO: move to a config file
EPOCHS      = 150
LR          = 1e-3
HIDDEN = [512, 512, 256, 256, 128]
SCHEDULER_PATIENCE = 15
VAL_DATA_PATH = "validation_data_3"
TEST_DATA_PATH = "test_data_3"
TRAINING_DATA_PATH   = "training_data_3"
NORM_PATH   = "orbit_norm_6.json"
MODEL_PATH  = "orbit_mlp_7.1.pt"

TRAINING_DATA_FILTER = "/orbit_train_part00[01234]*.npy"
# TRAINING_DATA_FILTER = "/orbit_train_part00[!01234]*.npy"
# --Time feature ffmapping

def fourier_time_features(t: np.ndarray, T: float = 0.23, n_harmonics: int = 4) -> np.ndarray:
    """
    Replace scalar time t with Fourier features.
    
    Args:
        t:            (N,) array of times in Gyr
        T:            base orbital period in Gyr
        n_harmonics:  number of sin/cos pairs
    
    Returns:
        (N, 2 * n_harmonics) array
    """
    features = []
    for k in range(1, n_harmonics + 1):
        features.append(np.sin(2 * np.pi * k * t / T))
        features.append(np.cos(2 * np.pi * k * t / T))
    return np.stack(features, axis=-1)

# ─── Dataset ─────────────────────────────────────────────────────────────────

def convert_features(X, t):
    r0 = np.sqrt(X[:, 0]**2 + X[:, 1]**2) # r0 = x0^2 + y0^2
    fourier = fourier_time_features(t)

    X = np.concatenate([X, r0[:, None], fourier], axis=1)
    return X

class OrbitDataset(Dataset):
    """
    Expects a data with shape (N, 10):
        columns 0-6:  x0, y0, z0, vx0, vy0, vz0, t   (inputs)
        columns 7-9:  x, y, z                          (outputs)
    All positions in parsecs, velocities in pc/Gyr, time in Gyr.

    Transforms to 15 input features input for MLP:
    Inputs: [x0, y0, z0, vx0, vy0, vz0, r0, sin1, cos1, sin2, cos2, sin3, cos3, sin4, cos4]
    Outputs: [x,y,z]
    """

    def __init__(self, folder_path: str, norm_stats, data_file_filter="/orbit_train_*.npy"):
        files = sorted(glob.glob(folder_path + data_file_filter))
        print("Files loaded: ", files)
        data  = np.concatenate([np.load(f) for f in files], axis=0)

        X = data[:, :6].astype(np.float32)
        y = (data[:, 7:] - data[:, :3]).astype(np.float32)
        t = data[:, 6].astype(np.float32)

        X = convert_features(X, t)

        # Normalize
        X = (X - norm_stats["X_mean"]) / norm_stats["X_std"]
        y = (y - norm_stats["y_mean"]) / norm_stats["y_std"]

        print("Inputs normalized, loading into torch.")
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Normalization ────────────────────────────────────────────────────────────

def compute_norm_stats(X, y, file_name=NORM_PATH):

    if os.path.exists(file_name):
        return load_norm_stats(file_name)
    
    print("Computing norm stats..")

    n = len(X)
    
    X_mean = X.sum(axis=0) / n
    y_mean = y.sum(axis=0) / n

    X_sq_sum = ((X - X_mean) ** 2).sum(axis=0)
    y_sq_sum = ((y - y_mean) ** 2).sum(axis=0)

    X_std = np.maximum(np.sqrt(X_sq_sum / n), 1e-8)
    y_std = np.maximum(np.sqrt(y_sq_sum / n), 1e-8)

    stats = {
        "X_mean": X_mean.tolist(),
        "X_std":  X_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std":  y_std.tolist(),
    }

    with open(file_name, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization stats to {file_name}")
    return stats


def compute_norm_stats_from_files(folder_paths: list, file_name=NORM_PATH) -> dict:
    """
    Compute mean and std for inputs and outputs from the full dataset.
    Save to JSON so inference can apply the same transform.
    """

    if os.path.exists(file_name):
        return load_norm_stats(file_name)
    
    print("Computing norm stats..")

    files = []
    for folder_path in folder_paths:
        files += glob.glob(folder_path + "/orbit_train_*.npy")
    files.sort()

    n = 0
    X_sum    = np.zeros(15, dtype=np.float64)
    y_sum    = np.zeros(3, dtype=np.float64)
    X_sq_sum = np.zeros(15, dtype=np.float64)
    y_sq_sum = np.zeros(3, dtype=np.float64)

    for f in files:
        data  = np.load(f,)
        X = data[:, :6].astype(np.float32)
        y = (data[:, 7:] - data[:, :3]).astype(np.float32)
        t = data[:, 6].astype(np.float32)
        X = convert_features(X, t)
        n += len(X)
        X_sum += X.sum(axis=0)
        y_sum += y.sum(axis=0)

    X_mean = X_sum / n
    y_mean = y_sum / n

    # second pass for std
    for f in files:
        data  = np.load(f,)
        X = data[:, :6].astype(np.float32)
        y = (data[:, 7:] - data[:, :3]).astype(np.float32)
        t = data[:, 6].astype(np.float32)
        X = convert_features(X, t)
        X_sq_sum += ((X - X_mean) ** 2).sum(axis=0)
        y_sq_sum += ((y - y_mean) ** 2).sum(axis=0)

    X_std = np.maximum(np.sqrt(X_sq_sum / n), 1e-8)
    y_std = np.maximum(np.sqrt(y_sq_sum / n), 1e-8)

    stats = {
        "X_mean": X_mean.tolist(),
        "X_std":  X_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std":  y_std.tolist(),
    }

    with open(file_name, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved normalization stats to {file_name}")
    return stats


def load_norm_stats(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise Exception(f"Couldn't find norm stats file at: {filepath}")
    with open(filepath) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


# ─── Model ───────────────────────────────────────────────────────────────────

class OrbitMLP(nn.Module):
    def __init__(self, hidden_sizes: list = HIDDEN):
        super().__init__()
        layers = []
        in_size = 15
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.SiLU())
            in_size = h
        layers.append(nn.Linear(in_size, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
# --- Residual Model

# class ResBlock(nn.Module):
#     def __init__(self, size):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(size, size),
#             nn.SiLU(),
#             nn.Linear(size, size),
#         )
#         self.act = nn.SiLU()

#     def forward(self, x):
#         return self.act(x + self.net(x))

# class OrbitMLP(nn.Module):
#     def __init__(self, hidden_sizes: list = HIDDEN):
#         super().__init__()
        
#         self.input_proj = nn.Linear(15, hidden_sizes[0])
        
#         blocks = []

#         if len(Counter(hidden_sizes)) > 1:
#             raise Exception("ResBlock requires all layers to be the same size. Input: ", hidden_sizes)

#         for size in hidden_sizes:
#             blocks.append(ResBlock(size))
#         self.blocks = nn.Sequential(*blocks)
        
#         self.output = nn.Linear(hidden_sizes[-1], 3)

#     def forward(self, x):
#         x = self.input_proj(x)
#         x = self.blocks(x)
#         return self.output(x)

# ─── Training ────────────────────────────────────────────────────────────────

def train(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True).float()
        y_batch = y_batch.to(DEVICE, non_blocking=True).float()

        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()
    
    return total_loss.item() / len(loader)

def train_full_gpu(model, X, y, optimizer, loss_fn, scaler):
    model.train()
    total_loss = 0.0
    n_batches = 0

    perm = torch.randperm(len(X), device=DEVICE)

    for i in range(0, len(X), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        X_batch = X[idx]
        y_batch = y[idx]

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE):
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)

        scaler.scale(loss).backward()       
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()


        total_loss += loss.detach()
        n_batches += 1

    return (total_loss / n_batches).item()

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

def evaluate_on_gpu(model, X, y, loss_fn, batch_size=BATCH_SIZE):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            predictions = model(X_batch)
            total_loss += loss_fn(predictions, y_batch).item()
    
    return total_loss / math.ceil(len(X) / batch_size)

# ─── Loss in parsecs ─────────────────────────────────────────────────────────

def loss_to_parsecs(mse_normalized: float, norm_stats: dict) -> float:
    y_std = norm_stats["y_std"]
    avg_std = float(np.mean(y_std))
    return np.sqrt(mse_normalized) * avg_std

# ─── Inference ───────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH, norm_path: str = NORM_PATH):
    """Load trained model and normalization stats for inference."""
    norm_stats = load_norm_stats(norm_path)

    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = OrbitMLP(hidden_sizes=checkpoint["hidden_sizes"]).to(DEVICE)
    model = torch.compile(model)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, norm_stats

def load_checkpoint(model, optimizer, scheduler, model_path: str = MODEL_PATH):
    """Resume training — restores optimizer and scheduler state."""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint.get("best_val_loss", float("inf"))

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
    start = time.time()

    r0 = np.sqrt(x0**2 + y0**2)
    fourier = fourier_time_features(np.array([t]))
    X = np.array([[x0, y0, z0, vx0, vy0, vz0]])
    X = np.concatenate([X, [[r0]], fourier], axis=1)
    X = (X - norm_stats["X_mean"]) / norm_stats["X_std"]


    X_tensor = torch.from_numpy(X).to(DEVICE)
    print(f"Data sent to gpu, {time.time() - start}")

    with torch.no_grad():
        y_norm = model(X_tensor).cpu().numpy()

    print(f"Predicted, {time.time() - start}")
    y = y_norm * norm_stats["y_std"] + norm_stats["y_mean"]
    return x0 + y[0, 0], y0 + y[0, 1], z0 + y[0, 2]


def predict_batch(model, norm_stats: dict, inputs: np.ndarray) -> np.ndarray:
    """
    Batch inference for N stars.

    Args:
        inputs: (N, 15)

    Returns:
        (N, 3) array of [x, y, z] in parsecs
    """
    r0 = np.sqrt(inputs[:, 0]**2 + inputs[:, 1]**2)
    fourier = fourier_time_features(inputs[:, 6])  # (N, 8)
    X = np.concatenate([inputs[:, :6], r0[:, None], fourier], axis=1)  # (N, 15)

    X = (X - norm_stats["X_mean"]) / norm_stats["X_std"]
    X_tensor = torch.from_numpy(X).to(DEVICE)

    with torch.no_grad():
        y_norm = model(X_tensor).cpu().numpy()

    displacement = y_norm * norm_stats["y_std"] + norm_stats["y_mean"]
    return inputs[:, :3] + displacement


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"{DEVICE=}")
    print(f"{BATCH_SIZE=}")
    print(f"{EPOCHS=}")
    print(f"{LR=}")
    print(f"{HIDDEN=}")
    print(f"{VAL_DATA_PATH=}")
    print(f"{TEST_DATA_PATH=}")
    print(f"{TRAINING_DATA_PATH=}")
    print(f"{NORM_PATH=}")
    print(f"{MODEL_PATH=}")


    torch.set_float32_matmul_precision("high")

    # --- dataset ---
    print("Loading dataset...")
    norm_stats = load_norm_stats(NORM_PATH)
    train_set = OrbitDataset(
        TRAINING_DATA_PATH,
        norm_stats,
        TRAINING_DATA_FILTER
    )
    test_set = OrbitDataset(TEST_DATA_PATH, norm_stats)
    val_set = OrbitDataset(VAL_DATA_PATH, norm_stats)

    # move full dataset to GPU
    train_X = train_set.X.to(DEVICE)
    train_y = train_set.y.to(DEVICE)

    val_X = val_set.X.to(DEVICE)
    val_y = val_set.y.to(DEVICE)

    test_X = test_set.X.to(DEVICE)
    test_y = test_set.y.to(DEVICE)

    scaler = GradScaler(device=DEVICE)

    # --- model ---
    if os.path.exists(MODEL_PATH):
        print("Loading Pre-trained model")
        model, _ = load_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=EPOCHS, eta_min=1e-6
        # )
        best_val_loss = load_checkpoint(
            model,
            optimizer,
            scheduler
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=50, T_mult=2, eta_min=1e-6
        # )
        # for group in optimizer.param_groups:
        #     group['lr'] = 1e-3

    else:
        model = OrbitMLP(hidden_sizes=HIDDEN).to(DEVICE)
        model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=EPOCHS, eta_min=1e-6
        # )
        best_val_loss = float("inf")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    loss_fn   = nn.MSELoss()    

    
    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_full_gpu(model, train_X, train_y, optimizer, loss_fn, scaler)
        val_loss = evaluate_on_gpu(model, val_X, val_y, loss_fn)

        scheduler.step(val_loss)
        # scheduler.step()

        train_pc = loss_to_parsecs(train_loss, norm_stats)
        val_pc   = loss_to_parsecs(val_loss,   norm_stats)

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss:.2e}  "
            f"val_loss={val_loss:.2e}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"{train_pc=}, {val_pc=}  "
            f"time={elapsed:.1f}s"
        )

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "hidden_sizes": HIDDEN,
                "norm_path": NORM_PATH,
                "best_val_loss": best_val_loss
            }, MODEL_PATH)

    # --- final test evaluation ---
    print("\nLoading best model for test evaluation...")
    model,_ = load_model(MODEL_PATH, NORM_PATH)

    test_loss = evaluate_on_gpu(model, test_X, test_y, loss_fn)
    test_pc   = loss_to_parsecs(test_loss, norm_stats)

    print(f"{test_pc=} parsecs")
    print(f"Target: 0.25 pc² MSE  (0.50 pc RMSE)")


if __name__ == "__main__":
    main()
