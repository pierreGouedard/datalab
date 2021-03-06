# Global import
import os
from numpy import nan
import pandas as pd
from pandas import isna
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional


def tansform_raw_data(
        df_raw_train: pd.DataFrame, df_raw_test: pd.DataFrame, index_col: str, weight_col: str, target_col: str,
        l_cat_cols: List[str], l_num_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transform raw data of the Kaggle Higgs dataset.

    The train dataset is used to standardized numerical features and filter columns based of covariance.

    Parameters
    ----------
    df_raw_train: DataFrame
        Train raw data.
    df_raw_test: DataFrame
        Test raw data.
    index_col: str
        Name of the index column.
    weight_col: str
        Name of the weight column.
    target_col: str
        Name of the target columns
    l_cat_cols: list
        List of name of categorical columns
    l_num_cols: list
        List of name of categorical columns

    Returns
    -------
    tuple
        composed of processed train, test and weights data.

    """
    # Set index
    df_raw_train, df_raw_test = df_raw_train.set_index(index_col), df_raw_test.set_index(index_col)

    # Extract weights
    df_weights = df_raw_train[[weight_col]]

    # Process train raw data
    df_train, scaler, l_filtered_num_cols = process_raw(
        df_raw_train[l_cat_cols + l_num_cols + [target_col]],
        l_col_cats=l_cat_cols,
        l_num_cols=l_num_cols,
        target_col=target_col,
        missing_value=-999,
    )

    # Process test raw data
    df_test, _, _ = process_raw(
        df_raw_test[l_cat_cols + l_num_cols],
        l_col_cats=l_cat_cols,
        l_num_cols=l_num_cols,
        missing_value=-999,
        scaler=scaler,
        l_filtered_num_cols=l_filtered_num_cols
    )

    return df_train, df_test, df_weights


def process_raw(
        df: pd.DataFrame, l_col_cats: List[str], l_num_cols: List[str], target_col: Optional[str] = None,
        missing_value: Optional[int] = None, scaler: Optional[StandardScaler] = None,
        l_filtered_num_cols: Optional[List[str]] = None, corr_thresh: Optional[float] = 0.95
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Process DataFrame.

    Build and fit model to standardize numerical columns if scaler not specified,  create list of filtered columns if
    l_filtered_num_cols is not specified.

    Parameters
    ----------
    df: DataFrame
        DataFrame to process.
    l_col_cats: list
        list of categrical columns.
    l_num_cols: list
        list of numerical columns.
    target_col: str
        Name of the target column.
    missing_value: int
        Value that take missing value in passed DataFrame.
    scaler: StandardScaler
        Optional standard scaler.
    l_filtered_num_cols: list
        Optional list of filtered numerical columns.
    corr_thresh: float
        Minimum value of correlation of columns with its predecessor up to which column is filtered out,
        value in [0, 1].

    Returns
    -------
    tuple
        Process DataFrame, Scaler and list of filtered numerical columns.

    """
    # Transform missing value into nan
    if missing_value is not None:
        df = df.replace(to_replace=missing_value, value=nan)

    # Create scaler
    if scaler is None:
        scaler = StandardScaler()
        ax_standardize = scaler.fit_transform(df[l_num_cols])

    else:
        ax_standardize = scaler.transform(df[l_num_cols])

    # Fit scaler and standardized numerical features
    df = df.assign(**{c: ax_standardize[:, i] for i, c in enumerate(l_num_cols)})

    # Remove too correlated features
    if l_filtered_num_cols is None:
        df_corr = df[l_num_cols].corr()
        l_filtered_num_cols = [df_corr.index[0]]
        for i in range(1, df_corr.shape[0]):
            if (df_corr.iloc[i].abs() > corr_thresh).iloc[:i].sum() == 0:
                l_filtered_num_cols.append(df_corr.index[i])

    # Gather selected cols
    if target_col is not None:
        l_cols = l_col_cats + l_filtered_num_cols + [target_col]
    else:
        l_cols = l_col_cats + l_filtered_num_cols

    return df[l_cols], scaler, l_filtered_num_cols
