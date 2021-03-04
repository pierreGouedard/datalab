"""Pipeline for activity pattern."""
# Global import
from kedro.pipeline import Pipeline, node

# Local import
from pipelines.higgs.prepare.transform import tansform_raw_data


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
    pass


def create_analyse_pipeline() -> Pipeline:
    """
    Create pipeline activity pattern.

    Returns
    -------
    Pipeline

    """
    pass