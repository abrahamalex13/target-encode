from collections import defaultdict
from functools import partial
import numpy as np


def update_target_conditionals_cv(x, y, idx_splits, target_prior_distribution):
    """
    Over data splits: given one data split, use "outside information" in others
    to update target conditionals (belief about) Y | X = x.

    Parameters
    ----------
    x : categorical-type variable
    y : target 
    idx_splits : list of CV fold assignments -- per fold element, row indexes
    target_prior_distribution : parameterized belief about target, X unknown

    Returns
    -------
    dict of cv_fold: target_conditionals mappings
    - One target_conditional mapping is level_x: target_summary

    """

    n_splits = len(idx_splits)

    cv_map = {
        "cv"
        + str(i): update_target_conditionals(
            x=x.drop(index=idx_splits[i]),
            y=y.drop(index=idx_splits[i]),
            target_prior_distribution=target_prior_distribution,
        )
        for i in range(n_splits)
    }

    cv_map["cv_mean"] = defaultdict(
        cv_map["cv0"].default_factory,
        {lvl: np.mean([cv_map[i][lvl] for i in cv_map]) for lvl in x.unique()},
    )

    return cv_map


def update_target_conditionals(x, y, target_prior_distribution):
    """
    Over levels x of categorical feature X: update target conditionals
    (belief about) Y | X = x.

    Parameters
    ----------
    x : categorical-type variable
    y : target
    target_prior_distribution : parameterized belief about target, X unknown

    Returns
    -------
    dict of level_x: target_summary mappings

    """

    # given never-observed X, expected Y derives directly from prior.
    # defaultdict succinctly handles requests for never-observed X
    # use of partial allows pickling, as opposed to within-fun defn.
    f_null_update = partial(update_target_belief, [], target_prior_distribution)

    levels_x_target_expected = defaultdict(f_null_update)

    for lvl in x.unique():
        levels_x_target_expected[lvl] = update_target_belief(
            y.loc[x == lvl], target_prior_distribution
        )

    return levels_x_target_expected


def update_target_belief(y, target_prior_distribution):
    """
    Update prior belief about target, using its realizations.

    Parameters
    ----------
    y : target (observations)
    target_prior_distribution : parameterized belief about target, X unknown

    Returns
    -------
    numeric, summary of target belief

    """

    n = len(y)

    if target_prior_distribution["family"] == "beta":

        if n > 0:

            n_Y_positives = sum(y)
            alpha_post = target_prior_distribution["alpha"] + n_Y_positives
            beta_post = target_prior_distribution["beta"] + n - n_Y_positives

        else:

            alpha_post = target_prior_distribution["alpha"]
            beta_post = target_prior_distribution["beta"]

        y_summary = alpha_post / (alpha_post + beta_post)

    elif target_prior_distribution["family"] == "normal":

        if n > 1:

            sample_mu = np.mean(y)

            # See Kevin Murphy's Probabilistic Machine Learning,
            # Gaussian-Gaussian model example.
            # As sample size rises, sample mean uncertainty decreases,
            # and deserves increasing weight
            var_sample_mu = np.var(y) / n
            weight_prior = var_sample_mu / (
                var_sample_mu + target_prior_distribution["variance_mu"]
            )

            mu_post = (
                weight_prior * target_prior_distribution["mu"]
                + (1 - weight_prior) * sample_mu
            )

        else:
            mu_post = target_prior_distribution["mu"]

        y_summary = mu_post

    else:
        y_summary = None

    return y_summary
