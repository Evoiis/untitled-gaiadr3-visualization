from orbit_mlp import compute_norm_stats_from_files, flogger

flogger.set_write_to_file(False)

compute_norm_stats_from_files(
    [
        "training_data_12",
    ],
    "orbit_norm_data_12.json"
)
