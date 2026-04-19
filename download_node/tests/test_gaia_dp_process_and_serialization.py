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
        original_ext_source_id=[np.nan, 7588],
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
        star_name=["", "Achernar"],
    ))


def proc() -> GaiaDataProcessor:
    return GaiaDataProcessor("data/")


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
        assert 1001 in stars.stars
        assert 1002 in stars.stars

    def test_positions_match(self):
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert stars.stars[1001].pos_x == pytest.approx(-0.5, abs=1e-6)
        assert stars.stars[1001].pos_y == pytest.approx(0.3, abs=1e-6)
        assert stars.stars[1001].pos_z == pytest.approx(0.1, abs=1e-6)

    def test_colors_match(self):
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert stars.stars[1001].color_r == 120
        assert stars.stars[1001].color_g == 200
        assert stars.stars[1001].color_b == 255

    def test_brightness_and_size_match(self):
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert stars.stars[1001].brightness == pytest.approx(0.4, abs=1e-5)
        assert stars.stars[1001].size == pytest.approx(2.5, abs=1e-5)

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

    def test_name_field_set_when_provided(self):
        df = make_processed_df()
        stars = deserialize(proc()._serialize_into_msg(df))
        assert stars.stars[1002].HasField("name")
        assert stars.stars[1002].name == "Achernar"

    def test_empty_dataframe_returns_valid_message(self):
        df = make_processed_df().iloc[0:0]
        blob = proc()._serialize_into_msg(df)
        stars = deserialize(blob)
        assert len(stars.stars) == 0

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
        ids = [s for s in stars.stars]
        assert 1001 in ids
        assert 1002 in ids

    def test_positions_are_finite(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        for star_id in stars.stars:
            assert np.isfinite(stars.stars[star_id].pos_x)
            assert np.isfinite(stars.stars[star_id].pos_y)
            assert np.isfinite(stars.stars[star_id].pos_z)

    def test_colors_in_valid_range(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        for star_id in stars.stars:
            assert 0 <= stars.stars[star_id].color_r <= 255
            assert 0 <= stars.stars[star_id].color_g <= 255
            assert 0 <= stars.stars[star_id].color_b <= 255

    def test_brightness_in_valid_range(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        for star_id in stars.stars:
            assert 0.0 <= stars.stars[star_id].brightness <= 1.0

    def test_size_positive(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        for star_id in stars.stars:
            assert stars.stars[star_id].size > 0

    def test_pipeline_calls_all_stages(self):
        """Verify all four compute stages are called."""
        p = proc()
        calls = []
        p._calculate_galactic_coordinates = lambda df: calls.append("coords")
        p._calculate_rgb_color             = lambda df: calls.append("color")
        p._calculate_star_brightness       = lambda df: calls.append("brightness")
        p._calculate_star_size             = lambda df: calls.append("size")
        p._match_star_names                = lambda df: calls.append("match")
        p._serialize_into_msg              = lambda df: calls.append("serialize")

        p.process_data(make_raw_df())
        assert calls == ["coords", "color", "brightness", "size", "match", "serialize"]

    def test_timestamp_present_in_output(self):
        df = make_raw_df()
        stars = deserialize(proc().process_data(df))
        assert stars.timestamp > 0
