from src.gaia_query import GaiaQueryWrapper, GaiaQueryParameters
from src.gaia_data_processor import GaiaDataProcessor

import time
import pandas as pd
import os

"""
Quick script to look at the data we're getting from a gaia query.
"""

file_name = "analysis_data_rv_500k_ranmod50_batch2.csv"

if os.path.exists(file_name):
    df = pd.read_csv(file_name)
else:
    df = None
    for batch_num in range(2):
        gqw = GaiaQueryWrapper(GaiaQueryParameters(
            guarantee_rad_velocity=True,
            ranset_mod=50,
            batch_num=batch_num
        ), wr_to_file=False)

        if batch_num == 0:
            df : pd.DataFrame = gqw.get_data()
        else:
            df = pd.concat([df, gqw.get_data()], ignore_index=True)
        
        # Wait to space out queries to Gaia Archive
        time.sleep(5)
        

    df.to_csv(file_name)



gdp = GaiaDataProcessor("data/")
gdp._calculate_cartesian_coordinates(df)
gdp._calculate_rgb_color(df)
gdp._calculate_star_brightness(df)
gdp._calculate_star_size(df)
gdp._match_star_names(df)

print(f"{df.isna().sum()=}")
print(f"{len(df)=}")

rgb = df[["color_r", "color_g", "color_b"]].values / 255.0

import matplotlib.pyplot as plt
plt.scatter(df["pos_x"], df["pos_y"], c=rgb, s=1)
p = plt.gca()
p.set_facecolor("black")
plt.title(file_name)
plt.show()
