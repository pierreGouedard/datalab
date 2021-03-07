# Global import
import numpy as np
import os

# Local import
from datalab.dataops.utils.names import KVName
from datalab.dataops.utils.driver import FileDriver


def send_xgb_submission(clf_xgb, df_test, index_col, threshold):
    """

    Parameters
    ----------
    clf_xgb
    df_test
    index_col
    threshold

    Returns
    -------

    """
    send_submission(clf_xgb, df_test.set_index(index_col), {'model': 'xgboost', 'threshold': threshold})


def send_rf_submission(clf_rf, df_test, index_col, threshold):
    """

    Parameters
    ----------
    clf_rf
    df_test
    index_col
    threshold

    Returns
    -------

    """
    send_submission(clf_rf, df_test.set_index(index_col), {'model': 'random_forest', 'threshold': threshold})


def send_submission(clf, df_test, d_model):
    """

    Parameters
    ----------
    clf
    df_test
    d_model

    Returns
    -------

    """
    # Build name of solution
    name_submission = KVName.from_dict(d_model).to_string()

    # Predict probas
    df_probas = clf.predict_proba(df_test)

    # Infer class using threshold
    df_probas = df_probas.loc[:, 's']\
        .sort_values(ascending=False)\
        .to_frame('Class')\
        .assign(
            rank=np.arange(len(df_test)), RankOrder=np.arange(len(df_test), 0, -1),
            Class=lambda df: np.where(df['Class'] > df['Class'].iloc[int(len(df) * d_model['threshold'])], 's', 'b')
        ) \
        .reset_index() \
        .loc[:, ['EventId', 'RankOrder', 'Class']]

    # Write submission in tmp dir
    driver = FileDriver('file_driver', "send submission to kaggle")
    tmp_file = driver.TempFile(prefix='kggl_higgs_', suffix='.csv')
    df_probas.to_csv(tmp_file.path)

    # Send submission using kaggle API
    bash = f'kaggle competitions submit -c higgs-boson -f "{tmp_file.path}" -m "{name_submission}"'
    os.system(bash)



