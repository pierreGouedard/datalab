"""Pipeline for activity pattern."""
# Global import
from kedro.pipeline import Pipeline, node

# Local import
from pipelines.higgs.prepare.transform import tansform_raw_data
from pipelines.higgs.fit.random_forest import select_and_fit_random_forest


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
                    "param_transform": "params:param_transform",
                    "param_transform_grid": "params:param_transform_grid",
                    "params_fold": "params:param_fold",
                    "scoring": "accuracy",
                },
                outputs=["higgs_train_transformed_data", "higgs_test_transformed_data", "higgs_weight_data"],
                name="transform_higgs_data",
            )
        ],
        tags=["higgs", "transform"],
    )


def create_analyse_pipeline() -> Pipeline:
    """
    Create pipeline activity pattern.

    Returns
    -------
    Pipeline

    """
    pass