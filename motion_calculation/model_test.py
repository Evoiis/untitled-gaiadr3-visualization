from orbit_mlp import load_model, OrbitDataset, evaluate_on_gpu, loss_to_parsecs
import torch
import torch.nn as nn

import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DATA_PATH = "test_data_3"
MODEL_PATH = "./orbit_mlp_9.pt"
NORM_PATH  = "./orbit_norm_6.json"
BATCH_SIZE = 102400

print("\nLoading model for test evaluation...")

torch.set_float32_matmul_precision("high")


model, norm_stats = load_model(MODEL_PATH, NORM_PATH)
# model = model.half()

test_set = OrbitDataset(TEST_DATA_PATH, norm_stats)
test_X = test_set.X.to(DEVICE)
test_y = test_set.y.to(DEVICE)

# test_X = test_X.half()
# test_y = test_y.half()

loss_fn = nn.MSELoss()

# warmup
with torch.no_grad():
    _ = model(test_X[:BATCH_SIZE])


# timed run
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    _ = model(test_X[:BATCH_SIZE])
torch.cuda.synchronize()
print(f"Single batch: {time.time()-t0:.4f}s")


start = time.time()
test_loss = evaluate_on_gpu(model, test_X, test_y, loss_fn, batch_size=BATCH_SIZE)

print(f"Time taken: {time.time() - start}")
print(f"{len(test_X)=}")

test_pc   = loss_to_parsecs(test_loss, norm_stats)

print(f"{test_pc=} parsecs")
print(f"Target:    0.25 pc² MSE  (0.50 pc RMSE)")

print(f"test_X device: {test_X.device}")
print(f"test_X dtype: {test_X.dtype}")
next(model.parameters()).device
next(model.parameters()).dtype
