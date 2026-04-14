"""
Orbit MLP - Surrogate model for galpy orbit integration
Predicts heliocentric XYZ position (parsecs) from initial conditions + time

"""

from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import torch
import yaml
import torch.nn as nn

from collections import Counter
from pprint import pformat
from file_logger import FileLogger
import numpy as np
import json
import time
import math
import os
import glob
import argparse

flogger = FileLogger()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset ---

def calc_time_fourier_features(t: np.ndarray, T: float = 0.23, n_harmonics: int = 4) -> np.ndarray:
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


def add_features(X, t):
    """
    Adds r0 and time fourier features
    """
    r0 = np.sqrt(X[:, 0]**2 + X[:, 1]**2) # r0 = x0^2 + y0^2
    fourier = calc_time_fourier_features(t)

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

    def __init__(self, folder_path: str, norm_stats, data_file_filter="/orbit_train_*.npy", use_half_precision=False):
        files = sorted(glob.glob(folder_path + data_file_filter))
        flogger.info(f"Files loaded: {files}")
        data  = np.concatenate([np.load(f) for f in files], axis=0)

        X = data[:, :6].astype(np.float32)
        y = (data[:, 7:] - data[:, :3]).astype(np.float32)
        t = data[:, 6].astype(np.float32)

        X = add_features(X, t)

        # Normalize
        X = (X - norm_stats["X_mean"]) / norm_stats["X_std"]
        y = (y - norm_stats["y_mean"]) / norm_stats["y_std"]

        flogger.info("Inputs normalized, loading into torch.")
        if use_half_precision:
            self.X = torch.from_numpy(X).to(torch.bfloat16)
            self.y = torch.from_numpy(y).to(torch.bfloat16)
        else:
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Normalization ---

def compute_norm_stats(X, y, file_name):

    if os.path.exists(file_name):
        return load_norm_stats(file_name)
    
    flogger.info("Computing norm stats..")

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
    flogger.info(f"Saved normalization stats to {file_name}")
    return stats


def compute_norm_stats_from_files(folder_paths: list, file_name) -> dict:
    """
    Compute mean and std for inputs and outputs from the full dataset.
    Save to JSON so inference can apply the same transform.
    """

    if os.path.exists(file_name):
        return load_norm_stats(file_name)
    
    flogger.info("Computing norm stats..")

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
        X = add_features(X, t)
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
        X = add_features(X, t)
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

    flogger.info(f"Saved normalization stats to {file_name}")
    return stats


def load_norm_stats(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise Exception(f"Couldn't find norm stats file at: {filepath}")
    with open(filepath) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


# --- MLP ---

class OrbitMLP(nn.Module):
    def __init__(self, hidden_sizes: list, config):
        super().__init__()
        layers = []
        in_size = 15
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            if "add_layer_norm" in config and config["add_layer_norm"]:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            in_size = h
        layers.append(nn.Linear(in_size, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- Residual MLP ---
class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_size),
            nn.Linear(in_size, out_size),
            nn.SiLU(),
            nn.LayerNorm(out_size),
            nn.Linear(out_size, out_size),
            nn.SiLU(),
        )
        # Only project if dimensions differ
        self.proj = nn.Linear(in_size, out_size, bias=False) if in_size != out_size else nn.Identity()

    def forward(self, x):
        return self.proj(x) + self.net(x)


class OrbitResidualMLP(nn.Module):
    def __init__(self, hidden_sizes: list, config):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(15, hidden_sizes[0]),
            nn.SiLU(),
        )

        blocks = []
        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            blocks.append(ResBlock(in_size, out_size))
        blocks.append(ResBlock(hidden_sizes[-1], hidden_sizes[-1]))
        self.blocks = nn.Sequential(*blocks)

        self.output = nn.Linear(hidden_sizes[-1], 3)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output(x)

# --- Fourier Feature MLP ---

class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_frequencies=64, sigma=1.0):
        super().__init__()
        # frozen random projection — not trained
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B  # (..., num_frequencies)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        # output dim = num_frequencies * 2

class OrbitFFMLP(nn.Module):
    def __init__(self, hidden_sizes: list, config):
        super().__init__()
        
        num_frequencies = config.get("fourier_frequencies", 64)
        sigma = config.get("fourier_sigma", 1.0)
        self.fourier = FourierFeatures(15, num_frequencies=num_frequencies, sigma=sigma)
        
        layers = []
        in_size = num_frequencies * 2
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            if config.get("add_layer_norm", False):
                layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            in_size = h
        layers.append(nn.Linear(in_size, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fourier(x)
        return self.net(x)

# --- SIREN MLP ---

class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, is_first=False, omega=30.0):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_dim, out_dim)
        self._init_weights(in_dim, is_first)

    def _init_weights(self, in_dim, is_first):
        with torch.no_grad():
            if is_first:
                # first layer uses wider uniform
                bound = 1.0 / in_dim
            else:
                bound = (6.0 / in_dim) ** 0.5 / self.omega
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class OrbitSiren(nn.Module):
    def __init__(self, hidden_sizes: list, omega=30.0):
        super().__init__()
        layers = [SirenLayer(15, hidden_sizes[0], is_first=True, omega=omega)]
        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(SirenLayer(in_size, out_size, omega=omega))
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_sizes[-1], 3)

    def forward(self, x):
        return self.output(self.net(x))

# --- Training ---

def train_with_dataloader(model, loader, optimizer, loss_fn, config) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=DEVICE, dtype=torch.bfloat16):
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)

        loss.backward()
        if config["use_gradient_clipping"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_max_norm"])
        optimizer.step()

        total_loss += loss.detach()
    
    return total_loss.item() / len(loader)

def train_with_full_dataset_on_gpu(model, X, y, optimizer, loss_fn, config):
    model.train()
    total_loss = 0.0
    n_batches = 0

    perm = torch.randperm(len(X), device=DEVICE)

    for i in range(0, len(X), config["batch_size"]):
        idx = perm[i:i+config["batch_size"]]

        X_batch = X[idx]
        y_batch = y[idx]

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE, dtype=torch.bfloat16):
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)

        loss.backward()
        if config["use_gradient_clipping"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_max_norm"])
        optimizer.step()

        total_loss += loss.detach()
        n_batches += 1

    return (total_loss / n_batches).item()

# --- Evaluate ---

def evaluate_with_dataloader(model, loader, loss_fn) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples

def evaluate_with_full_dataset_on_gpu(model, X, y, loss_fn, batch_size):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            predictions = model(X_batch)
            total_loss += loss_fn(predictions, y_batch).item()
    
    return total_loss / math.ceil(len(X) / batch_size)

# --- Inference ---

def load_checkpoint(model, optimizer, scheduler, model_path: str):
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

    X = np.array([[x0, y0, z0, vx0, vy0, vz0]])    
    X = add_features(X, np.array([t]))
    X = (X - norm_stats["X_mean"]) / norm_stats["X_std"]

    X_tensor = torch.from_numpy(X).to(DEVICE)
    flogger.info(f"Data sent to gpu, {time.time() - start}")

    with torch.no_grad():
        y_norm = model(X_tensor).cpu().numpy()

    flogger.info(f"Predicted, {time.time() - start}")
    y = y_norm * norm_stats["y_std"] + norm_stats["y_mean"]
    return x0 + y[0, 0], y0 + y[0, 1], z0 + y[0, 2]


def predict_batch(model, norm_stats: dict, inputs: np.ndarray) -> np.ndarray:
    """
    Batch inference for N stars.

    Args:
        inputs: (N, 7)

    Returns:
        (N, 3) array of [x, y, z] in parsecs
    """

    X = add_features(inputs[:, :6], inputs[:, 6])
    X = (X - norm_stats["X_mean"]) / norm_stats["X_std"]
    X_tensor = torch.from_numpy(X).to(DEVICE)

    with torch.no_grad():
        y_norm = model(X_tensor).float().cpu().numpy()

    displacement = y_norm * norm_stats["y_std"] + norm_stats["y_mean"]
    return inputs[:, :3] + displacement


# ─── Main ────────────────────────────────────────────────────────────────────

def loss_to_parsecs(mse_normalized: float, norm_stats: dict) -> float:
    y_std = norm_stats["y_std"]
    avg_std = float(np.mean(y_std))
    return np.sqrt(mse_normalized) * avg_std

def loss_to_parsecs_huber(huber_loss_val, norm_stats):
    """
    Approximates parsecs from the Huber scalar.
    WARNING: Only accurate when error < delta (0.1).
    """
    # 1. Huber (for small errors) = 0.5 * MSE
    # So, MSE = 2 * Huber
    mse_normalized = 2 * huber_loss_val
    
    # 2. Convert normalized MSE to parsecs
    avg_std = np.mean(norm_stats["y_std"])
    return math.sqrt(mse_normalized) * avg_std

def load_config(config_file_path: str):
    with open(config_file_path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    return data

def load_data(train_set, val_set, test_set):
    train_X = train_set.X.to(DEVICE)
    train_y = train_set.y.to(DEVICE)

    val_X = val_set.X.to(DEVICE)
    val_y = val_set.y.to(DEVICE)

    test_X = test_set.X.to(DEVICE)
    test_y = test_set.y.to(DEVICE)

    return train_X, train_y, val_X, val_y, test_X, test_y

def load_dataloaders(config, train_set, val_set, test_set):
    n_cpus = os.cpu_count()
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,  num_workers=max(n_cpus - 1, 1), pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=config["batch_size"], shuffle=False, num_workers=max(n_cpus - 1, 1), pin_memory=True, prefetch_factor=4, persistent_workers=True)
    test_loader  = DataLoader(test_set,  batch_size=config["batch_size"], shuffle=False, num_workers=max(n_cpus - 1, 1), pin_memory=True, prefetch_factor=4, persistent_workers=True)
    
    return  train_loader, val_loader, test_loader

def init_scheduler(config, optimizer):
    if config["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=config["patience"], min_lr=config["min_learning_rate"]
        )
    elif config["scheduler"] == "cosanneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["min_learning_rate"]
        )
    elif config["scheduler"] == "cosannealwarmrestart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config["warm_restart_epochs"], T_mult=config["warm_restart_cycle_mult"], eta_min=config["min_learning_rate"]
        )
    else:
        raise Exception(f"Unexpected scheduler choice in config: {config["scheduler"]=}")
    
    return scheduler

def init_model(config, hidden_sizes):

    if "model_type" not in config:
        model = OrbitMLP(hidden_sizes=hidden_sizes, config=config).to(DEVICE)

    if config["model_type"] == "residual":
        model = OrbitResidualMLP(hidden_sizes=hidden_sizes, config=config).to(DEVICE)
    elif config["model_type"] == "fourier":
        model = OrbitFFMLP(hidden_sizes, config).to(DEVICE)
    elif config["model_type"] == "siren":
        model = OrbitSiren(hidden_sizes).to(DEVICE)
    else:
        model = OrbitMLP(hidden_sizes=hidden_sizes, config=config).to(DEVICE)
    
    return model

def load_model_from_file(config):
    """Load trained model."""
    torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(config["model_name"], map_location=DEVICE)

    flogger.info(
        f"Checkpoint loaded: {pformat(checkpoint["hidden_sizes"])}" +
        f", {pformat(checkpoint["norm_path"])}"
    )

    model = init_model(config, checkpoint["hidden_sizes"])

    model = torch.compile(model)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_model(config):
    flogger.info("Loading model")

    best_val_loss = float("inf")

    if os.path.exists(config["model_name"]):
        flogger.info("Found existing model, loading from file.")
        model = load_model_from_file(config)
    else:
        flogger.info("Creating a new model from scratch.")

        model = init_model(config, config["hidden_layers"])
        
        model = torch.compile(model)

    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["start_learning_rate"])
    elif config["optimizer"] == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["start_learning_rate"])
    else:
        raise Exception(f"Unexpected optimizer choice in config: {config["optimizer"]=}")


    scheduler = init_scheduler(config, optimizer)

    if os.path.exists(config["model_name"]):
        best_val_loss = load_checkpoint(
            model,
            optimizer,
            scheduler,
            config["model_name"]
        )

        if "override_learning_rate" in config and config["override_learning_rate"]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config["start_learning_rate"]
            scheduler = init_scheduler(config, optimizer)
            best_val_loss = float('inf')
    
    return model, scheduler, optimizer, best_val_loss

def run_training_run(config):
    flogger.info("Loading dataset...")
    norm_stats = load_norm_stats(config["norm_path"])
    train_set = OrbitDataset(
        config["training_data_path"],
        norm_stats,
        config["training_data_filter"],
        config["use_half_precision"]
    )
    test_set = OrbitDataset(config["test_data_path"], norm_stats)
    val_set = OrbitDataset(config["val_data_path"], norm_stats)


    if config["use_dataloader"]:
        train_loader, val_loader, test_loader = load_dataloaders(config, train_set, val_set, test_set)
    else:
        train_X, train_y, val_X, val_y, test_X, test_y = load_data(train_set, val_set, test_set)
        

    if "loss_fn" in config and config["loss_fn"] == "huber":
        loss_fn = nn.HuberLoss(delta=config["huber_delta"])
    else:
        loss_fn = nn.MSELoss()

    model, scheduler, optimizer, best_val_loss = load_model(config)
    

    total_params = sum(p.numel() for p in model.parameters())
    flogger.info(f"Model parameters: {total_params:,}")
    
    flogger.info("\nStarting training...\n")
    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        if config["use_dataloader"]:
            train_loss = train_with_dataloader(model, train_loader, optimizer, loss_fn, config)
            val_loss = evaluate_with_dataloader(model, val_loader, loss_fn)
        else:
            train_loss = train_with_full_dataset_on_gpu(
                model,
                train_X,
                train_y,
                optimizer,
                loss_fn,
                config
            )
            val_loss = evaluate_with_full_dataset_on_gpu(model, val_X, val_y, loss_fn, config["batch_size"])

        if config["scheduler"] == "plateau":
            scheduler.step(val_loss)
        elif config["scheduler"] in ["cosanneal", "cosannealwarmrestart"]:
            scheduler.step()            
        else:
            raise Exception(f"Unexpected scheduler choice in config: {config["scheduler"]=}")
        

        if "loss_fn" in config and config["loss_fn"] == "huber":
            train_pc = loss_to_parsecs_huber(train_loss, norm_stats)
            val_pc   = loss_to_parsecs_huber(val_loss, norm_stats)
        else:
            train_pc = loss_to_parsecs(train_loss, norm_stats)
            val_pc   = loss_to_parsecs(val_loss,   norm_stats)

        elapsed = time.time() - t0

        flogger.info(
            f"Epoch {epoch}/{config["epochs"]}  "
            f"train_loss={train_loss:.2e}  "
            f"val_loss={val_loss:.2e}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"{train_pc=}, {val_pc=}  "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "hidden_sizes": config["hidden_layers"],
                "norm_path": config["norm_path"],
                "best_val_loss": best_val_loss
            }, config["model_name"])

    # --- final test evaluation ---
    flogger.info("\nLoading best model for test evaluation...")
    model = load_model_from_file(config)

    if config["use_dataloader"]:
        test_loss = evaluate_with_dataloader(model, test_loader, loss_fn)
    else:
        test_loss = evaluate_with_full_dataset_on_gpu(model, test_X, test_y, loss_fn, config["batch_size"])
    if "loss_fn" in config and config["loss_fn"] == "huber":
        test_pc = loss_to_parsecs_huber(test_loss, norm_stats)
    else:
        test_pc = loss_to_parsecs(test_loss, norm_stats)

    flogger.info(f"{test_pc=} parsecs")

def main():
    torch.set_float32_matmul_precision("high")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()

    config = load_config(args.config_file)

    flogger.set_folder_and_file(
        "logs",
        f"{time.strftime("%y_%m_%d_%H_%M_%S")}_{config["model_name"][:-3]}.txt"
    )

    flogger.info(
        f"Loaded config: {pformat(config)}"
    )

    runs = config.pop("runs")
    run_list = list(runs.keys())
    run_list.sort()
    for run in run_list:
        flogger.info(f"Starting run: {run}")        
        run_training_run(config | runs[run])

if __name__ == "__main__":
    main()
