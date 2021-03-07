# Global import
import os
import pandas as pd
import sys

# Local import
from datalab.dataops.ml_operations.clf_model import ClfSelector, Classifier


# Parameters for xgboost
params_xgb = {
    'nthread': 2, 'n_jobs': 2, 'objective': "binary:logistic", "verbosity": 0, "colsample_bytree": 0.8,
    "subsample": 0.8, 'eta': 0.1, "max_depth": 8, 'min_child_weight': 250, 'alpha': 0, 'lambda': 0,
    'min_child_leaf': 30, 'gamma':  1, "n_estimators": 500
}
params_xgb_grid = {
    'min_child_leaf': [40, 20], 'gamma': [1, 0.5], "n_estimators": [500, 800]
}


def select_and_fit_xgboost(
        df_train, df_weights, l_cat_cols, l_num_cols, target_col, param_mdl, param_mdl_grid, param_transform,
        param_transform_grid, params_fold, scoring
):
    """

    Parameters
    ----------
    df_train
    df_weights
    l_cat_cols
    l_num_cols
    target_col
    param_mdl
    param_mdl_grid
    param_transform
    param_transform_grid
    params_fold
    scoring

    Returns
    -------

    """
    param_transform['num_cols'] = list(set(l_num_cols).intersection(df_train.columns))
    param_transform['cat_cols'] = list(set(l_cat_cols).intersection(df_train.columns))
    param_transform['target_col'] = target_col

    cs = ClfSelector(
        df_data=df_train[param_transform['cat_cols'] + param_transform['num_cols']],
        df_weights=df_weights,
        model_name='xgb',
        param_mdl=param_mdl,
        param_mdl_grid=param_mdl_grid,
        param_transform=param_transform,
        param_transform_grid=param_transform_grid,
        params_fold=params_fold,
        scoring=scoring
    )

    classifier = cs.fit().get_classifier()
    _, _, d_scores = classifier.evaluate(cs.fold_manager.df_train, cs.fold_manager.df_test)

    return classifier, d_scores
