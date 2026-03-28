import pytest
import numpy as np
import pandas as pd

from gaia_data_processor import GaiaDataProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal DataFrame with only the columns needed for a test."""
    return pd.DataFrame(kwargs)


def proc() -> GaiaDataProcessor:
    return GaiaDataProcessor("data/")


# ---------------------------------------------------------------------------
# _calculate_cartesian_coordinates
# ---------------------------------------------------------------------------

class TestCartesianCoordinates:

    def test_sun_direction_galactic_center(self):
        """RA=0, dec=0, parallax=1000 mas (1 pc) → (1, 0, 0)."""
        df = make_df(ra=[0.0], dec=[0.0], parallax=[1000.0])
        proc()._calculate_cartesian_coordinates(df)
        assert pytest.approx(df["pos_x"][0], abs=1e-6) == 1.0
        assert pytest.approx(df["pos_y"][0], abs=1e-6) == 0.0
        assert pytest.approx(df["pos_z"][0], abs=1e-6) == 0.0

    def test_north_pole(self):
        """dec=90 → star directly above, z = distance."""
        df = make_df(ra=[0.0], dec=[90.0], parallax=[1000.0])
        proc()._calculate_cartesian_coordinates(df)
        assert pytest.approx(df["pos_z"][0], abs=1e-6) == 1.0
        assert pytest.approx(df["pos_x"][0], abs=1e-6) == 0.0
        assert pytest.approx(df["pos_y"][0], abs=1e-6) == 0.0

    def test_distance_scaling(self):
        """10 pc star (parallax=100 mas) should be 10x further than 1 pc star."""
        df = make_df(ra=[0.0, 0.0], dec=[0.0, 0.0], parallax=[1000.0, 100.0])
        proc()._calculate_cartesian_coordinates(df)
        assert pytest.approx(df["pos_x"][1], rel=1e-6) == df["pos_x"][0] * 10

    def test_ra_rad_and_dec_rad_columns_created(self):
        df = make_df(ra=[45.0], dec=[30.0], parallax=[500.0])
        proc()._calculate_cartesian_coordinates(df)
        assert "ra_rad" in df.columns
        assert "dec_rad" in df.columns
        assert pytest.approx(df["ra_rad"][0]) == np.deg2rad(45.0)
        assert pytest.approx(df["dec_rad"][0]) == np.deg2rad(30.0)

    def test_output_columns_present(self):
        df = make_df(ra=[10.0], dec=[20.0], parallax=[200.0])
        proc()._calculate_cartesian_coordinates(df)
        for col in ("pos_x", "pos_y", "pos_z"):
            assert col in df.columns

    def test_radec_to_xyz_alpha_centauri(self):
        df = make_df(ra=[219.9], dec=[-60.8], parallax=[746.26865671])

        proc()._calculate_cartesian_coordinates(df)

        assert df["pos_x"][0] == pytest.approx(-0.502, abs=1e-3)
        assert df["pos_y"][0] == pytest.approx(-0.419, abs=1e-3)
        assert df["pos_z"][0] == pytest.approx(-1.170, abs=1e-3)


# ---------------------------------------------------------------------------
# _calculate_rgb_color
# ---------------------------------------------------------------------------

class TestRGBColor:

    def _run(self, bp_rp_values):
        df = make_df(bp_rp=bp_rp_values)
        proc()._calculate_rgb_color(df)
        return df

    def test_output_columns_present(self):
        df = self._run([0.0])
        for col in ("color_r", "color_g", "color_b"):
            assert col in df.columns

    def test_hot_star_is_blue(self):
        """bp_rp=-0.5 (hottest) → low R, high B."""
        df = self._run([-0.5])
        assert df["color_b"][0] == pytest.approx(255, abs=1)
        assert df["color_r"][0] == pytest.approx(0, abs=2)

    def test_cool_star_is_red(self):
        """bp_rp=5.0 (coolest) → full R, zero B."""
        df = self._run([5.0])
        assert df["color_r"][0] == pytest.approx(255, abs=1)
        assert df["color_b"][0] == pytest.approx(0, abs=1)

    def test_white_point(self):
        """bp_rp  = 2.0 --> R=255, G=255, B=255."""
        # white_point in bp_rp space: norm = white_point → bp_rp = 2.5 - 0.5 = 2.0
        df = self._run([2.0])
        assert df["color_r"][0] == pytest.approx(255, abs=2)
        assert df["color_g"][0] == pytest.approx(255, abs=2)
        assert df["color_b"][0] == pytest.approx(255, abs=2)

    def test_values_clipped_to_uint8_range(self):
        df = self._run([-0.5, 0.0, 2.0, 4.0, 5.0])
        for col in ("color_r", "color_g", "color_b"):
            assert (df[col] >= 0).all(), f"{col} has negative values"
            assert (df[col] <= 255).all(), f"{col} exceeds 255"

    def test_out_of_range_bp_rp_clipped(self):
        """Values outside [-0.5, 5.0] should not crash and should equal the boundary values."""
        df_low  = self._run([-99.0])
        df_high = self._run([99.0])
        df_bound_low  = self._run([-0.5])
        df_bound_high = self._run([5.0])
        for col in ("color_r", "color_g", "color_b"):
            assert df_low[col][0]  == pytest.approx(df_bound_low[col][0],  abs=1)
            assert df_high[col][0] == pytest.approx(df_bound_high[col][0], abs=1)

    def test_monotonic_red_channel(self):
        """R should increase monotonically with bp_rp (hotter → less red)."""
        bp_rp_vals = np.linspace(-0.5, 5.0, 50)
        df = self._run(bp_rp_vals)
        assert (np.diff(df["color_r"].values) >= -0.5).all()

    def test_monotonic_blue_channel(self):
        """B should decrease monotonically with bp_rp (cooler → less blue)."""
        bp_rp_vals = np.linspace(-0.5, 5.0, 50)
        df = self._run(bp_rp_vals)
        assert (np.diff(df["color_b"].values) <= 0.5).all()


# ---------------------------------------------------------------------------
# _calculate_star_brightness
# ---------------------------------------------------------------------------

class TestStarBrightness:
    
    def _base_df(self, lum=None, mag=None, n=1):
        length = len(lum) if lum is not None else len(mag) if mag is not None else n
        return make_df(
            lum_flame=lum if lum is not None else [np.nan] * length,
            phot_g_mean_mag=mag if mag is not None else [15.0] * length,
        )

    def test_output_columns_present(self):
        df = self._base_df(lum=[1.0], mag=[15.0])
        proc()._calculate_star_brightness(df)
        assert "brightness" in df.columns

    def test_brightness_normalized_range(self):
        df = self._base_df(lum=[1e-3, 1.0, 1e3, 1e6], mag=[15.0] * 4)
        proc()._calculate_star_brightness(df)
        assert (df["brightness"] >= 0.0).all()
        assert (df["brightness"] <= 1.0).all()

    def test_brighter_lum_higher_brightness(self):
        df = self._base_df(lum=[0.01, 1.0, 100.0], mag=[15.0] * 3)
        proc()._calculate_star_brightness(df)
        b = df["brightness"].values
        assert b[0] < b[1] < b[2]

    def test_fallback_mag_used_when_lum_nan(self):
        """When lum_flame is NaN, phot_g_mean_mag should drive brightness."""
        df = make_df(
            lum_flame=[np.nan, np.nan],
            phot_g_mean_mag=[10.0, 18.0],   # brighter mag → higher brightness
        )
        proc()._calculate_star_brightness(df)
        assert df["brightness"][0] > df["brightness"][1]

    def test_lum_takes_priority_over_mag(self):
        """A high-lum star should outshine a low-mag (bright apparent) star."""
        df = make_df(
            lum_flame=[1e5, np.nan],
            phot_g_mean_mag=[18.0, 5.0],
        )
        proc()._calculate_star_brightness(df)
        assert df["brightness"][0] > df["brightness"][1]

    def test_no_nans_in_output(self):
        df = make_df(
            lum_flame=[np.nan, 1.0, np.nan],
            phot_g_mean_mag=[12.0, 15.0, 18.0],
        )
        proc()._calculate_star_brightness(df)
        assert not df["brightness"].isna().any()


# ---------------------------------------------------------------------------
# _calculate_star_size
# ---------------------------------------------------------------------------

class TestStarSize:

    def _base_df(self, lum=None, teff=None, bp_rp=None, n=1):
        def _col(val, length):
            return val if val is not None else [np.nan] * length

        length = len(lum) if lum is not None else len(teff) if teff is not None else len(bp_rp) if bp_rp is not None else n

        return make_df(
            lum_flame=_col(lum, length),
            teff_gspphot=_col(teff, length),
            bp_rp=_col(bp_rp, length) if bp_rp is not None else [1.0] * length,
        )

    def test_output_column_present(self):
        df = self._base_df(lum=[1.0])
        proc()._calculate_star_size(df)
        assert "size" in df.columns

    def test_size_positive(self):
        df = self._base_df(lum=[1e-3, 1.0, 1e6])
        proc()._calculate_star_size(df)
        assert (df["size"] > 0).all()

    def test_higher_lum_larger_size(self):
        df = self._base_df(lum=[0.01, 1.0, 1000.0])
        proc()._calculate_star_size(df)
        s = df["size"].values
        assert s[0] < s[1] < s[2]

    def test_teff_fallback_hotter_larger(self):
        """Without lum_flame, hotter teff → larger size."""
        df = make_df(
            lum_flame=[np.nan, np.nan],
            teff_gspphot=[3500.0, 20000.0],
            bp_rp=[2.0, 0.0],
        )
        proc()._calculate_star_size(df)
        assert df["size"][1] > df["size"][0]

    def test_bp_rp_fallback_bluer_larger(self):
        """Without lum_flame or teff, lower bp_rp (bluer/hotter) → larger size."""
        df = make_df(
            lum_flame=[np.nan, np.nan],
            teff_gspphot=[np.nan, np.nan],
            bp_rp=[4.0, 0.0],
        )
        proc()._calculate_star_size(df)
        assert df["size"][1] > df["size"][0]

    def test_lum_takes_priority_over_teff(self):
        df = make_df(
            lum_flame=[1e5, np.nan],
            teff_gspphot=[np.nan, 50000.0],
            bp_rp=[1.0, 1.0],
        )
        proc()._calculate_star_size(df)
        assert df["size"][0] > df["size"][1]

    def test_no_nans_in_output(self):
        df = make_df(
            lum_flame=[np.nan, 1.0, np.nan],
            teff_gspphot=[5000.0, np.nan, np.nan],
            bp_rp=[1.5, 1.0, 2.0],
        )
        proc()._calculate_star_size(df)
        assert not df["size"].isna().any()

    def test_size_minimum_is_positive(self):
        """Even the dimmest/coolest star should have size > 0."""
        df = self._base_df(lum=[1e-3], teff=[3000.0], bp_rp=[5.0])
        proc()._calculate_star_size(df)
        assert df["size"][0] > 0
