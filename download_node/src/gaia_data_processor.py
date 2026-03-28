import pandas as pd
import numpy as np
import time

import star_data_pb2

class GaiaDataProcessor():

    def __init__(self, data_folder_path: str):
        iau_data = pd.read_csv(data_folder_path + "iau_stars.csv")
        iau_data = iau_data[iau_data['HIP'].notna()]
        self.hip_to_name = dict(zip(iau_data["HIP"].dropna().astype(int), iau_data["Proper Names"]))

    def process_data(self, df: pd.DataFrame):
        self._calculate_cartesian_coordinates(df)
        self._calculate_rgb_color(df)
        self._calculate_star_brightness(df)
        self._calculate_star_size(df)
        self._match_star_names(df)

        return self._serialize_into_msg(df)

    def _serialize_into_msg(self, df: pd.DataFrame):
        stars = star_data_pb2.Stars()
        stars.timestamp = int(time.time())

        df[["color_r", "color_g", "color_b"]] = df[["color_r", "color_g", "color_b"]].astype(int)

        for row in df.itertuples(index=False):
            star = stars.stars.add()
            star.id = row.source_id
            star.pos_x = row.pos_x
            star.pos_y = row.pos_y
            star.pos_z = row.pos_z
            star.color_r = row.color_r
            star.color_g = row.color_g
            star.color_b = row.color_b
            star.brightness = row.brightness
            star.size = row.size
            if "star_name" in row:
                star.name = row.star_name

        return stars.SerializeToString()
    
    def _match_star_names(self, df: pd.DataFrame):       
        df["star_name"] = df["original_ext_source_id"].map(self.hip_to_name).fillna("")

    def _calculate_cartesian_coordinates(self, df: pd.DataFrame):
        df["ra_rad"] = np.deg2rad(df["ra"].values)
        df["dec_rad"] = np.deg2rad(df["dec"].values)

        parsecs = 1000 / df["parallax"]

        df["pos_x"] = parsecs * np.cos(df["dec_rad"]) * np.cos(df["ra_rad"])
        df["pos_y"] = parsecs * np.cos(df["dec_rad"]) * np.sin(df["ra_rad"])
        df["pos_z"] = parsecs * np.sin(df["dec_rad"])

    def _calculate_rgb_color(self, df: pd.DataFrame):
        # bp_rp -> color
        # red -> white -> blue
        # -0.5 --> 2.5 -> 5.0

        bp_rp = np.clip(df["bp_rp"], -0.5, 5.0)
        
        white_point = 2.5 / 5.5
        norm = (bp_rp + 0.5)/ 5.5

        low_norm = norm / white_point
        high_norm = (norm - white_point)/(1 - white_point)
        
        df["color_r"] = np.where(
            norm < white_point,
            low_norm * 255,
            255
        )

        df["color_g"] = np.where(
            norm < white_point,
            100 + (low_norm * 155),
            255 - high_norm * 225
        )

        df["color_b"] = np.clip(
            np.where(
                norm <= white_point,
                255,
                255 - high_norm * 255 * 2
            ),
            0,
            255
        )

    def _calculate_star_brightness(self, df: pd.DataFrame):
        # Primary: lum_flame (solar luminosities, log-distributed)
        # Fallback: phot_g_mean_mag (apparent mag, inverted)
        
        brightness = np.where(
            df["lum_flame"].notna(),
            np.log10(np.clip(df["lum_flame"], 1e-3, 1e6)),
            -(df["phot_g_mean_mag"] - 20.0) / 20.0   
        )
        
        # Normalize to [0, 1]
        b_min, b_max = -3.0, 6.0
        brightness_norm = np.clip((brightness - b_min) / (b_max - b_min), 0.0, 1.0)
        
        df["brightness"] = brightness_norm

    def _calculate_star_size(self, df: pd.DataFrame):
        # Primary: teff_gspphot — hotter stars are physically larger on the main sequence
        # but giants/supergiants break this, so lum_flame is a better size proxy if available
        # Fallback: bp_rp (inverted, blue stars tend larger/hotter)

        size = np.where(
            df["lum_flame"].notna(),
            np.log10(np.clip(df["lum_flame"], 1e-3, 1e6)),   # same log L — giants naturally big
            np.where(
                df["teff_gspphot"].notna(),
                np.log10(np.clip(df["teff_gspphot"], 3000, 50000)) - np.log10(3000),  # [0, ~1.2]
                np.clip(1.0 - (df["bp_rp"] + 0.5) / 5.5, 0.0, 1.0)  # inverted norm
            )
        )

        # Normalize to [0, 1]
        s_min, s_max = -3.0, 6.0
        size_norm = np.clip((size - s_min) / (s_max - s_min), 0.0, 1.0)

        # Scale to point sprite size — tune these once you see it
        df["size"] = 1.0 + size_norm * 9.0   # [1, 10] px, placeholder range
