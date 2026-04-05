from orbit_mlp import load_model, OrbitDataset, evaluate_on_gpu, loss_to_parsecs
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DATA_PATH = "test_data"
MODEL_PATH = "./orbit_mlp_5.pt"
NORM_PATH  = "./orbit_norm_5.json"

print("\nLoading model for test evaluation...")

torch.set_float32_matmul_precision("high")


model, norm_stats = load_model(MODEL_PATH, NORM_PATH)

test_set = OrbitDataset(TEST_DATA_PATH, norm_stats)
test_X = test_set.X.to(DEVICE)
test_y = test_set.y.to(DEVICE)

loss_fn = nn.MSELoss()

test_loss = evaluate_on_gpu(model, test_X, test_y, loss_fn)
test_pc   = loss_to_parsecs(test_loss, norm_stats)

print(f"{test_pc=} parsecs")
print(f"Target:    0.25 pc² MSE  (0.50 pc RMSE)")
