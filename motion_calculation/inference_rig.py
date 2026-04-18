from orbit_mlp import (
    load_model_from_file,
    predict_batch
)

class InferenceRig():
    """
        Interface to abstract mlp loading and prediction.
    """

    def __init__(self, config):
