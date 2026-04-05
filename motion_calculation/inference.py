# import numpy as np
# import torch
# import time

# from orbit_mlp import load_model, predict, predict_batch

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL_PATH = "./orbit_mlp_5.pt"
# NORM_PATH  = "./orbit_norm_5.json"

# def main():
#     torch.set_float32_matmul_precision("high")
#     model, norm_stats = load_model(MODEL_PATH, NORM_PATH)   

#     start = time.time()

#     x, y, z = predict(
#         model, norm_stats,
#         x0=7987.65472121391, 
#         y0=3.45292700609862, 
#         z0=5.2936880458663, 
#         vx0=95959.3485136876, 
#         vy0=307137.832071721, 
#         vz0=153507.337653083,
#         t=-3
#     )

#     # 1.238552797296416763e+01,3.452915798934854230e+00,-1.547418472654159949e+01
#     print("TT: ", time.time() - start)

#     print("Single:", x, y, z)

#     # inputs = np.array([
#     #     [8000, 0, 20, 0, 220, 10, 0.1],
#     #     [8000, 100, 30, -10, 210, 5, 0.2],
#     # ], dtype=np.float32)

#     # outputs = predict_batch(model, norm_stats, inputs)

#     # print("Batch:\n", outputs)

# if __name__ == "__main__":
#     main()
