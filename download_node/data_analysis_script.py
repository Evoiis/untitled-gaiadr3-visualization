from src.gaia_query import GaiaQueryWrapper, GaiaQueryParameters
from src.gaia_data_processor import GaiaDataProcessor
import pandas as pd
import os

"""
Quick script to look at the data we're getting from a gaia query.
"""

file_name = "analysis_data_rv_5k_ranmod50_x.csv"

if os.path.exists(file_name):
    df = pd.read_csv(file_name)
else:
    gqw = GaiaQueryWrapper(GaiaQueryParameters(
        guarantee_rad_velocity=True,
        n_stars=5000
    ), wr_to_file=False)
    df : pd.DataFrame = gqw.get_data()
    df.to_csv(file_name)

print(f"{df.isna().sum()=}")

print(f"{len(df)=}")




gdp = GaiaDataProcessor("data/")
gdp._calculate_cartesian_coordinates(df)
gdp._calculate_rgb_color(df)

print(df[["pos_x", "bp_rp"]].corr())
rgb = df[["color_r", "color_g", "color_b"]].values / 255.0

import matplotlib.pyplot as plt
plt.scatter(df["pos_x"], df["pos_y"], c=rgb, s=1)
p = plt.gca()
p.set_facecolor("black")
plt.title(file_name)
plt.show()
