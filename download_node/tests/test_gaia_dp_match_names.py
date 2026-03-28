import pandas as pd
import numpy as np

from gaia_data_processor import GaiaDataProcessor

DATA_FOLDER = "data/"

def make_proc() -> GaiaDataProcessor:
    return GaiaDataProcessor(data_folder_path=DATA_FOLDER)

def make_df(hip_ids: list) -> pd.DataFrame:
    return pd.DataFrame({"original_ext_source_id": hip_ids})


class TestMatchStarNames:

    def test_output_column_created(self):
        df = make_df([7588])
        make_proc()._match_star_names(df)
        assert "star_name" in df.columns

    def test_acamar_hip_13847(self):
        df = make_df([13847])
        make_proc()._match_star_names(df)
        assert df["star_name"][0] == "Acamar"

    def test_unknown_hip_maps_to_empty_string(self):
        df = make_df([99999999])
        make_proc()._match_star_names(df)
        assert df["star_name"][0] == ""

    def test_nan_hip_maps_to_empty_string(self):
        df = make_df([np.nan])
        make_proc()._match_star_names(df)
        assert df["star_name"][0] == ""

    def test_mixed_known_unknown_nan(self):
        df = make_df([7588, 99999999, np.nan, 13847])
        make_proc()._match_star_names(df)
        assert df["star_name"][0] == "Achernar"
        assert df["star_name"][1] == ""
        assert df["star_name"][2] == ""
        assert df["star_name"][3] == "Acamar"

    def test_no_nans_in_output(self):
        df = make_df([7588, 99999999, np.nan])
        make_proc()._match_star_names(df)
        assert not df["star_name"].isna().any()

    def test_empty_dataframe(self):
        df = make_df([])
        make_proc()._match_star_names(df)
        assert "star_name" in df.columns
        assert len(df) == 0
