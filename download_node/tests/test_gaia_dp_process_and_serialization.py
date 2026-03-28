import time
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

import star_data_pb2
from gaia_data_processor import GaiaDataProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raw_df(**kwargs) -> pd.DataFrame:
    """Minimal raw input DataFrame mimicking Gaia query output."""
    defaults = dict(
        source_id=[1001, 1002],
        ra=[10.0, 20.0],
        dec=[5.0, -10.0],
        parallax=[100.0, 200.0],
        bp_rp=[1.0, 3.0],
        phot_g_mean_mag=[12.0, 15.0],
        lum_flame=[1.0, np.nan],
        teff_gspphot=[5500.0, np.nan],
    )
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def make_processed_df() -> pd.DataFrame:
    """DataFrame that already has all computed columns — for serialization tests."""
    return pd.DataFrame(dict(
        source_id=[1001, 1002],
        pos_x=[-0.5, 0.8],
        pos_y=[0.3, -0.6],
        pos_z=[0.1, 0.2],
        color_r=[120, 255],
        color_g=[200, 180],
        color_b=[255, 0],
        brightness=[0.4, 0.7],
        size=[2.5, 6.0],
    ))


def proc() -> GaiaDataProcessor:
    return GaiaDataProcessor()


def deserialize(blob: bytes) -> star_data_pb2.Stars:
    stars = star_data_pb2.Stars()
    stars.ParseFromString(blob)
    return stars


# ---------------------------------------------------------------------------
# _serialize_into_msg
# ---------------------------------------------------------------------------

class TestSerializeIntoMsg:

    def test_returns_bytes(self):
        df = make_processed_df()
        result = proc()._serialize_into_msg(df)
        assert isinstance(result, bytes)

    def test_deserializes_without_error(self):
        df = make_processed_df()
        blob = proc()._serialize_into_msg(df)
        stars = deserialize(blob)
        assert len(stars.stars) == 2

    def test_star_count_matches_dataframe(self):
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert len(stars.stars) == len(df)

    def test_star_ids_match(self):
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert stars.stars[0].id == 1001
        assert stars.stars[1].id == 1002

    def test_positions_match(self):
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert stars.stars[0].pos_x == pytest.approx(-0.5, abs=1e-6)
        assert stars.stars[0].pos_y == pytest.approx(0.3, abs=1e-6)
        assert stars.stars[0].pos_z == pytest.approx(0.1, abs=1e-6)

    def test_colors_match(self):
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert stars.stars[0].color_r == 120
        assert stars.stars[0].color_g == 200
        assert stars.stars[0].color_b == 255

    def test_brightness_and_size_match(self):
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert stars.stars[0].brightness == pytest.approx(0.4, abs=1e-5)
        assert stars.stars[0].size == pytest.approx(2.5, abs=1e-5)

    def test_timestamp_is_set(self):
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert stars.timestamp > 0

    def test_timestamp_is_recent(self):
        before = int(time.time())
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        after = int(time.time())
        assert before <= stars.timestamp <= after

    def test_name_field_not_set_by_default(self):
        """Optional name field should be unset when not provided."""
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert not stars.stars[0].HasField("name")

    def test_empty_dataframe_returns_valid_message(self):
        df = make_processed_df().iloc[0:0]  # empty, same columns
        blob = proc()._serialize_into_msg(df)
        stars = deserialize(blob)
        assert len(stars.stars) == 0

    def test_single_star(self):
        df = make_processed_df().iloc[[0]]
        stars = deserialize(proc()._serialize_into_msg(df))
        assert len(stars.stars) == 1
        assert stars.stars[0].id == 1001


# ---------------------------------------------------------------------------
# process_data (integration)
# ---------------------------------------------------------------------------

class TestProcessData:

    def test_returns_bytes(self):
        df = make_raw_df()
        result = proc().process_data(df)
        assert isinstance(result, bytes)

    def test_deserializes_without_error(self):
        df = make_raw_df()
        blob = proc().process_data(df)
        stars = deserialize(blob)
        assert len(stars.stars) == 2

    def test_star_count_matches_input(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        assert len(stars.stars) == len(df)

    def test_ids_preserved(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        ids = [s.id for s in stars.stars]
        assert 1001 in ids
        assert 1002 in ids

    def test_positions_are_finite(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        for star in stars.stars:
            assert np.isfinite(star.pos_x)
            assert np.isfinite(star.pos_y)
            assert np.isfinite(star.pos_z)

    def test_colors_in_valid_range(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        for star in stars.stars:
            assert 0 <= star.color_r <= 255
            assert 0 <= star.color_g <= 255
            assert 0 <= star.color_b <= 255

    def test_brightness_in_valid_range(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        for star in stars.stars:
            assert 0.0 <= star.brightness <= 1.0

    def test_size_positive(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        for star in stars.stars:
            assert star.size > 0

    def test_pipeline_calls_all_stages(self):
        """Verify all four compute stages are called."""
        p = proc()
        calls = []
        p._calculate_cartesian_coordinates = lambda df: calls.append("cartesian")
        p._calculate_rgb_color             = lambda df: calls.append("color")
        p._calculate_star_brightness       = lambda df: calls.append("brightness")
        p._calculate_star_size             = lambda df: calls.append("size")
        p._serialize_into_msg              = lambda df: calls.append("serialize") or b""

        p.process_data(make_raw_df())
        assert calls == ["cartesian", "color", "brightness", "size", "serialize"]

    def test_timestamp_present_in_output(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        assert stars.timestamp > 0
