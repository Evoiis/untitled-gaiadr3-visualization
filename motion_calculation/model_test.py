from orbit_mlp import load_model_from_file, OrbitDataset, evaluate_with_full_dataset_on_gpu, loss_to_parsecs, load_config, load_norm_stats, flogger
import torch
import torch.nn as nn

import time
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "configs/config_20.yaml"

print("\nLoading model for test evaluation...")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")

flogger.set_write_to_file(False)
config = load_config(CONFIG_FILE)

# --- INPUT -------------------------------------------------
TEST_DATA_PATH = "./prev_data/test_data_3"
NORM_PATH = config["norm_path"]
MODEL_PATH = config["model_name"]

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "./prev_models/" + config["model_name"]

# OVERRIDE
# NORM_PATH = "orbit_norm_6.json"
# MODEL_PATH = "orbit_mlp_20_300.pt"

# ^^^ INPUT ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print(f"\nRunning model test on: {MODEL_PATH}")
print(f"Using norms: {NORM_PATH}\n")
model = load_model_from_file(config)
norm_stats = load_norm_stats(NORM_PATH)

test_set = OrbitDataset(TEST_DATA_PATH, norm_stats)
test_X = test_set.X.to(DEVICE)
test_y = test_set.y.to(DEVICE)

loss_fn = nn.MSELoss()

# warmup
# with torch.no_grad():
#     _ = model(test_X[:config["batch_size"]])


start = time.time()
test_loss = evaluate_with_full_dataset_on_gpu(model, test_X, test_y, loss_fn, batch_size=config["batch_size"])

print(f"Time taken: {time.time() - start}")
print(f"{len(test_X)=}")

test_pc = loss_to_parsecs(test_loss, norm_stats)

print(f"{test_pc=} parsecs")

import matplotlib.pyplot as plt
import numpy as np

def plot_xyz_comparison(pred, target, title="XYZ Comparison", n_samples=500):
    """
    pred, target: (N, 3) numpy arrays in parsecs
    """
    idx = np.random.choice(len(pred), min(n_samples, len(pred)), replace=False)
    p = pred[idx]
    t = target[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    for i, axis in enumerate(['x', 'y', 'z']):
        ax = axes[i]
        lim = max(np.abs(t[:, i]).max(), np.abs(p[:, i]).max())
        ax.scatter(t[:, i], p[:, i], alpha=0.3, s=5)
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
        ax.set_xlabel(f"True {axis} (pc)")
        ax.set_ylabel(f"Pred {axis} (pc)")
        ax.set_title(f"{axis} — RMSE: {np.sqrt(((p[:, i] - t[:, i])**2).mean()):.2f} pc")
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.close()
