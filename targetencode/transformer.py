import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from targetencode import update_target_conditionals_cv


class TargetEncodeTransformer(BaseEstimator, TransformerMixin):
    """
    Update beliefs about target (Y) conditional distributions -- 
    Y | X = x, where X takes categorical levels x.

    Then overwrite X categorical levels with summary of updated Y | X = x.
    Iterate over categorical features in a dataset.

    When fitting this transformer (estimator), _beware data leakage_:
    do not use an observation's Y to target-encode its X. 
    Follow cross-validation (CV) approach: target-encode one split 
    using "outside information" in the other splits.
    """

    def __init__(
        self,
        features,
        n_cv_splits,
        target_prior_distribution,
        dir_save=None,
        name_save=None,
    ):

        self.features = features
        self.n_cv_splits = n_cv_splits
        self.target_prior_distribution = target_prior_distribution
        self.dir_save = dir_save
        self.name_save = name_save

    def fit(self, X, y):

        lbl_splits = np.random.choice(
            range(self.n_cv_splits), size=X.shape[0], replace=True
        )
        self.idx_splits = [
            np.where(lbl_splits == i)[0] for i in range(self.n_cv_splits)
        ]

        self.features_cv_map = {}
        for var in self.features:

            self.features_cv_map[var] = update_target_conditionals_cv(
                X[var], y, self.idx_splits, self.target_prior_distribution
            )

        self.is_next_transform_by_cv_fold = True

        return self

    def transform(self, X):
        """
        Procedure varies by transformer context/state.
        If immediately following `fit`, follow CV splits when overwriting X.
        If general transform case, use CV mean.
        """
        return self.get_transform_fun()(X)

    def get_transform_fun(self):
        if self.is_next_transform_by_cv_fold:
            return self.transform_by_cv_fold
        else:
            return self.transform_via_cv_mean

    def transform_by_cv_fold(self, X):

        for var in self.features_cv_map:

            # cannot stepwise overwrite categorical (str) with numeric encoding
            x_encoded = np.repeat(-1000.0, repeats=X.shape[0])

            for i in range(self.n_cv_splits):

                idx_split = self.idx_splits[i]
                mapping = self.features_cv_map[var]["cv" + str(i)]
                x_encoded[idx_split] = X[var].iloc[idx_split].map(mapping)

            X.loc[:, var] = x_encoded

        self.is_next_transform_by_cv_fold = False

        self.feature_names_out = X.columns

        return X

    def transform_via_cv_mean(self, X):

        for var in self.features_cv_map:

            mapping = self.features_cv_map[var]["cv_mean"]
            X.loc[:, var] = X[var].map(mapping)

        self.feature_names_out = X.columns

        return X

    def save(self):
        pickle.dump(self, open(self.dir_save + self.name_save + ".pkl", "wb"))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out
