"""
Quick and dirty script to test models with test data

- Timing taken data is not reliable!
    - First run will be slower due to cold start
    - Time taken print is only for a very rough idea of speed
"""

from orbit_mlp import load_model_from_file, OrbitDataset, evaluate_with_full_dataset_on_gpu, loss_to_parsecs, load_config, load_norm_stats
import torch
import torch.nn as nn

import argparse
import time
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
test_data_folders = [
    "prev_data/test_data_3",
    "test_data_12",
    "data/dataset_13S/test_data",
    "data/dataset_14_20p/test_data",
    "data/dataset_15_300k/test_data",
]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")

# python model_test.py <config>
# optional --model_name <model_name>
parser = argparse.ArgumentParser()
parser.add_argument("config_file")
parser.add_argument("--model_name", required=False)
args = parser.parse_args()

config = load_config(args.config_file)

if args.model_name:
    config["model_name"] = args.model_name

norm_path = config["norm_path"]

if not os.path.exists(config["model_name"]):
    config["model_name"] = "./prev_models/" + config["model_name"]

if not os.path.exists(norm_path):
    norm_path = "./norms/" + norm_path


print(f"\nLoading model {config['model_name']=} for test evaluation...")
model = load_model_from_file(config)
norm_stats = load_norm_stats(config)

for test_folder in test_data_folders:

    print(f"\n\nLoading test data from: {test_folder}")
    test_set = OrbitDataset(test_folder, norm_stats)
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

    print(f"{test_pc} parsecs test error")

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_xyz_comparison(pred, target, title="XYZ Comparison", n_samples=500):
#     """
#     pred, target: (N, 3) numpy arrays in parsecs
#     """
#     idx = np.random.choice(len(pred), min(n_samples, len(pred)), replace=False)
#     p = pred[idx]
#     t = target[idx]

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     fig.suptitle(title)

#     for i, axis in enumerate(['x', 'y', 'z']):
#         ax = axes[i]
#         lim = max(np.abs(t[:, i]).max(), np.abs(p[:, i]).max())
#         ax.scatter(t[:, i], p[:, i], alpha=0.3, s=5)
#         ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
#         ax.set_xlabel(f"True {axis} (pc)")
#         ax.set_ylabel(f"Pred {axis} (pc)")
#         ax.set_title(f"{axis} — RMSE: {np.sqrt(((p[:, i] - t[:, i])**2).mean()):.2f} pc")
#         ax.set_aspect('equal')

#     plt.tight_layout()
#     plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
#     plt.close()
