from src.gaia_query import GaiaQueryWrapper, GaiaQueryParameters
from src.gaia_data_processor import GaiaDataProcessor

from time import time
"""
Quick script to plot the data we're getting from a gaia query.
"""

gqw = GaiaQueryWrapper(
    GaiaQueryParameters(
        n_stars_per_batch=1,
        random_set_modulo=50
    )
)

start = time()

df = gqw.get_data(8)

print("Query time taken: ", time() - start)

gdp = GaiaDataProcessor("data/")
gdp._calculate_galactic_coordinates(df)
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
# plt.title(file_name)
plt.show()
