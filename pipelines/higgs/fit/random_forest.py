# Global import

# Local import
from datalab.dataops.ml_operations.clf_model import ClfSelector, Classifier


def select_and_fit_random_forest(
        df_train, df_weights, param_mdl, param_mdl_grid, param_transform, param_transform_grid, params_fold, scoring
):
    """

    Parameters
    ----------
    df_train
    df_weights
    param_mdl
    param_mdl_grid
    param_transform
    param_transform_grid
    params_fold
    scoring

    Returns
    -------

    """
    cs = ClfSelector(
        df_data=df_train,
        df_weights=df_weights,
        model_name='rf',
        param_mdl=param_mdl,
        param_mdl_grid=param_mdl_grid,
        param_transform=param_transform,
        param_transform_grid=param_transform_grid,
        params_fold=params_fold,
        scoring=scoring
    )

    classifier = cs.fit().save_classifier(model_path).get_classifier()
    _, _, d_scores = classifier.evaluate(cs.fold_manager.df_train, cs.fold_manager.df_test)

    # # Compute submission ? Now ?
    # df_submission = classifier.predict(cs.fold_manager.df_test)\
    #     .to_frame('Class')\
    #     .assign(
    #         Class=lambda x: classifier.feature_builder.target_encoder.inverse_transform(x.Class),
    #         RankOrder=lambda x: range(len(x))
    #     )\
    #     .reset_index()\
    #     .loc[:, ['EventId', 'RankOrder', 'Class']]\
    #     .to_csv(submission_path, index=None)


