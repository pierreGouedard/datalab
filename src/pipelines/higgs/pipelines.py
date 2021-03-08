"""Pipeline for activity pattern."""
# Global import
from kedro.pipeline import Pipeline, node

# Local import
from pipelines.higgs.prepare.transform import tansform_raw_data
from pipelines.higgs.fit.random_forest import select_and_fit_random_forest
from pipelines.higgs.fit.xgboost import select_and_fit_xgboost
from pipelines.higgs.submit.submit import send_rf_submission, send_xgb_submission


def create_transform_pipeline() -> Pipeline:
    """
    Create pipeline activity pattern.

    Returns
    -------
    Pipeline

    """
    return Pipeline(
        [
            node(
                tansform_raw_data,
                inputs={
                    "df_raw_train": "higgs_train_raw_data",
                    "df_raw_test": "higgs_test_raw_data",
                    "index_col": "params:index_col",
                    "weight_col": "params:weight_col",
                    "target_col": "params:target_col",
                    "l_cat_cols": "params:cat_col",
                    "l_num_cols": "params:num_col"
                },
                outputs=["higgs_train_transformed_data", "higgs_test_transformed_data", "higgs_weight_data"],
                name="transform_higgs_data",
            )
        ],
        tags=["higgs", "transform"],
    )


def create_fit_pipeline() -> Pipeline:
    """
    Create pipeline activity pattern.

    Returns
    -------
    Pipeline

    """
    return Pipeline(
        [
            node(
                select_and_fit_random_forest,
                inputs={
                    "df_train": "higgs_train_transformed_data",
                    "df_weights": "higgs_weight_data",
                    "l_cat_cols": "params:cat_col",
                    "l_num_cols": "params:num_col",
                    "target_col": "params:target_col",
                    "param_mdl": "params:rf_param",
                    "param_mdl_grid": "params:rf_param_grid",
                    "param_transform": "params:param_rf_transform",
                    "param_transform_grid": "params:param_rf_transform_grid",
                    "params_fold": "params:param_fold",
                    "scoring": "params:higgs_scoring",
                },
                outputs=["rf_model", "rf_scores"],
                tags=["higgs", "fit", 'rf'],
                name="fit_random_forest",
            ),
            node(
                select_and_fit_xgboost,
                inputs={
                    "df_train": "higgs_train_transformed_data",
                    "df_weights": "higgs_weight_data",
                    "l_cat_cols": "params:cat_col",
                    "l_num_cols": "params:num_col",
                    "target_col": "params:target_col",
                    "param_mdl": "params:xgb_param",
                    "param_mdl_grid": "params:xgb_param_grid",
                    "param_transform": "params:param_xgb_transform",
                    "param_transform_grid": "params:param_xgb_transform_grid",
                    "params_fold": "params:param_fold",
                    "scoring": "params:higgs_scoring",
                },
                outputs=["xgb_model", "xgb_scores"],
                tags=["higgs", "fit", 'xgb'],
                name="fit_xgboost",
            )
        ],
    )


def create_submit_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                send_rf_submission,
                inputs={
                    "clf_rf": "rf_model",
                    "df_test": "higgs_test_transformed_data",
                    "index_col": "params:index_col",
                    "threshold": "params:higgs_proba_threshold",
                },
                outputs=None,
                tags=["higgs", "submit", 'rf'],
                name="submit_random_forest",
            ),
            node(
                send_xgb_submission,
                inputs={
                    "clf_xgb": "xgb_model",
                    "df_test": "higgs_test_transformed_data",
                    "index_col": "params:index_col",
                    "threshold": "params:higgs_proba_threshold",
                },
                outputs=None,
                tags=["higgs", "submit", 'xgb'],
                name="submit_xgboost",
            )
        ],
    )


def create_analyse_pipeline() -> Pipeline:
    """
    Create pipeline activity pattern.

    Returns
    -------
    Pipeline

    """
    pass