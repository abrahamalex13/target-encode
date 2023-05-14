import pytest
import pandas as pd
import numpy as np

from targetencode import TargetEncodeTransformer, update_target_conditionals_cv


@pytest.fixture()
def data_target_encode():
    """
    Test data record successful history and troubled present for MLB teams. 
    Data structure imply test of successful target (CV) target encode: 
    encode success during troubled regimes, troubled during success regimes.
    """
    return pd.read_csv("./tests/data/example_target_encode.csv")


@pytest.fixture()
def target_encoder(data_target_encode):

    target_encoder = TargetEncodeTransformer(
        features=["x1"],
        n_cv_splits=2,
        target_prior_distribution={"family": "beta", "alpha": 0, "beta": 0},
    )

    target_encoder.idx_splits = [np.arange(0, 6, 1), 6 + np.arange(0, 6, 1)]

    target_encoder.features_cv_map = {}
    for var in target_encoder.features:

        target_encoder.features_cv_map[var] = update_target_conditionals_cv(
            data_target_encode[var],
            data_target_encode["y"],
            target_encoder.idx_splits,
            target_encoder.target_prior_distribution,
        )

    target_encoder.is_next_transform_by_cv_fold = True

    return target_encoder
