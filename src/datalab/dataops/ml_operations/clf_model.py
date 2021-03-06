# Global imports
import itertools
import logging
import pandas as pd
import numpy as np
import dill as pickle
import os
from sklearn.metrics import precision_score, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, \
    confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import xgboost
from scipy.sparse import spmatrix
from typing import Dict, Any, Optional

# Local import
from datalab.dataops.utils.names import KVName
from datalab.dataops.ml_operations.features import FoldManager


class ClfSelector(object):
    """
    The ClfSelector is an object that supervise the selection of hyper parameter of feature transformation and
    model training.

    """
    allowed_score = ['precision', 'accuracy', 'balanced_accuracy', 'f1_score', 'roc']
    allowed_model = ['dt', 'rf', 'xgb', 'yala']

    def __init__(
            self, df_data: pd.DataFrame, model_name: str, param_mdl: Dict[str, Any], param_mdl_grid: Dict[str, Any],
            param_transform: Dict[str, Any], param_transform_grid: Dict[str, Any], params_fold: Dict[str, Any],
            scoring: str, df_weights: Optional[pd.DataFrame], weight_arg: Optional[str] = None,
            scoring_arg: str = 'scoring', path_intermediate_result=None

    ) -> None:
        """

        Parameters
        ----------
        df_data : pandas.DataFrame
            DataFrame composed of document text and label.

        model_name : str
            Name of the model to use to classify documents.

        param_transform : dict
            Dictionary of fixed parameter for transforming input data.

        param_transform_grid : dict
            Dictionary of parameter for transforming input data to select using a grid search.

        param_mdl : dict
            Dictionary of fixed parameter for model.

        param_mdl_grid : dict
            Dictionary of parameter for model to select using a grid search.

        params_fold : dict
            Dictionary of parameter to use to make the cross validation of features builder and model.

        scoring : str
            Scoring method to use to evaluate models and models builder.

        """
        assert model_name in self.allowed_model, "Choose model among {}".format(self.allowed_model)
        assert scoring in self.allowed_score, "Choose scoring among {}".format(self.allowed_score)

        # Core parameter classifier
        self.model_name = model_name
        self.scoring = scoring
        self.weight_arg, self.scoring_arg = weight_arg, scoring_arg

        # Parameters of models builder and model
        self.param_transform, self.param_transform_grid = param_transform, param_transform_grid
        self.param_mdl, self.param_mdl_grid = param_mdl, param_mdl_grid

        # Core attribute of classifier
        self.fold_manager = FoldManager(df_data, df_weights=df_weights, **params_fold)
        self.is_fitted, self.model, self.d_search = False, None, None

    def fit(self):
        """
        Find the optimal combination of hyper parameter of models building and model through grid search. Fit the
        entire etl with the selected parameters.

        """
        # Fit model and models builder using K-fold cross validation
        self.d_search = self.grid_search()
        self.model = self.__fit_all()
        self.is_fitted = True

        return self

    def grid_search(self):
        """
        Perform a cross validation of hyper parameters of the features building routine and the model used for
        classification. For given parameter of models builder, we retain the parameter of prediction model that get
        the best score. The parameter of models builder with the higher score will be chosen, along with the best model
        associated.

        """

        if len(self.param_transform_grid) <= 1 and len(self.param_mdl_grid) <= 1:
            return {0: {"params_feature": self.param_transform, "param_mdl": self.param_mdl}, 'best_key': 0}

        d_search, best_score = {}, 0.
        for i, cross_values in enumerate(itertools.product(*self.param_transform_grid.values())):

            # Reset fold_manager to allow fit of new models builder
            self.fold_manager.reset()

            # Update params of grid search
            d_features_grid_instance = self.param_transform.copy()
            d_features_grid_instance.update(dict(zip(self.param_transform_grid.keys(), cross_values)))
            d_search[i] = {'params_feature': d_features_grid_instance.copy()}

            # Inform about Current params
            logging.info("[FEATURE]: {}".format(KVName.from_dict(d_features_grid_instance).to_string()))

            # Fit model and keep params of best model associated with current models's parameters
            best_mdl_params, best_mdl_score = self.grid_search_mdl(d_features_grid_instance)
            d_search[i].update({'param_mdl': best_mdl_params, 'best_score': best_mdl_score})

            # Keep track of best association models builder / model
            if best_score < best_mdl_score:
                d_search['best_key'] = i

        logging.info('Optimal parameters found are {}'.format(d_search[d_search['best_key']]))

        return d_search

    def grid_search_mdl(self, d_feature_params):
        """
        Perform a cross validation of hyper parameter of the logistic regression for given word embedding parameter.

        Parameters
        ----------
        d_feature_params : dict
            Dictionary of parameters for building features.

        Returns
        -------
        tuple
            A tuple containing a dictionary of parameters of the best model and the score of the best model
        """
        best_mdl_params, best_mdl_score = None, 0.
        for cross_values in itertools.product(*self.param_mdl_grid.values()):

            # Update params of logistic regression model
            d_mdl_grid_instance = self.param_mdl.copy()
            d_mdl_grid_instance.update(dict(zip(self.param_mdl_grid.keys(), cross_values)))

            # Instantiate model and cross validate the hyper parameter
            model, args = get_model(self.model_name, d_mdl_grid_instance)
            l_errors = []

            for d_train, d_val in self.fold_manager.generate_folds(d_feature_params):
                # Fit and predict
                if 'input_shape' in args.keys():
                    args['input_shape'] = d_train['X'].shape[1]

                if 'w' in d_train.keys() and self.weight_arg is not None:
                    args[self.weight_arg] = d_train['w']

                model.fit(d_train['X'], d_train['y'], **args)
                score = get_score(self.scoring, model.predict(d_val['X']), d_val['y'])

                # Get error
                l_errors.append(score)

            # Average errors and update best params if necessary
            mu = np.mean(l_errors)
            logging.info('[MODEL]: {} | score: {}'.format(KVName.from_dict(d_mdl_grid_instance).to_string(), mu))

            if best_mdl_score < mu:
                best_mdl_params, best_mdl_score = d_mdl_grid_instance.copy(), mu

        return best_mdl_params, best_mdl_score

    def __fit_all(self):
        """
        Fit models builder and model based on optimal hyper parameter found using.

        :return: trained model

        """

        # Reset data_manager to allow fit of we
        self.fold_manager.reset()

        # Recover optimal parameters
        d_feature_params = self.d_search[self.d_search['best_key']]['params_feature']
        d_model_params = self.d_search[self.d_search['best_key']]['param_mdl']
        d_train = self.fold_manager.get_all_train_data(d_feature_params)

        # Instantiate model and fit it
        model, kwargs = get_model(self.param_transform, d_model_params)
        d_eval = self.fold_manager.get_eval_data(d_feature_params)

        if 'input_shape' in kwargs.keys():
            kwargs['input_shape'] = d_train['X'].shape[1]

        if d_train.get('w', None) is not None:
            kwargs[self.weight_arg] = d_train['w']

        if d_train.get('s', None) is not None:
            kwargs[self.scoring_arg] = d_train['s']

        if 'args' in d_feature_params.keys():
            kwargs.update({arg: d_train[arg] for arg in d_feature_params['args']})

        if d_eval is not None:
            kwargs['eval_set'] = d_eval

        model.fit(d_train['X'], d_train['y'], **kwargs)

        return model

    def get_classifier(self):

        # Get best params for feature and model
        d_param_feature = self.d_search[self.d_search['best_key']]['params_feature']
        d_param_model = self.d_search[self.d_search['best_key']]['param_mdl']

        return Classifier(self.model, self.fold_manager.data_transformer, d_param_model, d_param_feature)

    def save_classifier(self, path):
        """
        Save core element of the documetn classifier.

        Parameters
        ----------
        path : str
            path toward the location where classifier shall be saved.

        Returns
        -------

        """
        # Get and pickle classifier
        with open(path, 'wb') as handle:
            pickle.dump(self.get_classifier(), handle)

        return self

    def save_data(self, path, name_train, name_test):
        """
        Save data used to select and fit the classifier.

        Parameters
        ----------
        path : str
            path toward the location where classifier shall be saved.
        name_train : str
            name of file containing train data.
        name_test : str
            name of file containing test data.

        Returns
        -------

        """

        path_train, path_test = os.path.join(path, name_train), os.path.join(path, name_test)
        self.fold_manager.df_train.to_hdf(path_train, key=name_train.split('.')[0], mode='w')
        self.fold_manager.df_test.to_hdf(path_test, key=name_test.split('.')[0], mode='w')

        return self


class Classifier(object):
    """
    The Classifier is an object that ready to use to classify.

    """

    def __init__(self, model, data_transformer, param_model, param_transform):
        """

        Parameters
        ----------
        model : object
            Fitted model to use to classify documents.

        data_transformer : src.model.feature.FeatureBuilder
            Fitted model to use to vectorize text of documents.

        param_transform : dict
            Dictionary of fixed parameter for building features.

        param_model : dict
            Dictionary of fixed parameter for classification model.


        """

        # Core parameter classifier
        self.model = model
        self.data_transformer = data_transformer
        self.param_model = param_model
        self.param_transform = param_transform

    @staticmethod
    def from_path(path):
        with open(path, 'rb') as handle:
            dc = pickle.load(handle)

        return Classifier(**dc.__dict__)

    def predict(self, df):
        """
        Predict target for feature in df.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------

        """
        features = self.data_transformer.transform(df)
        preds = self.model.predict(features)

        return pd.Series(preds, index=df.index, name="prediction")

    def predict_proba(self, df, **kwargs):
        """
        Predict probabilities over target space for feature in df.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------

        """
        features = self.data_transformer.transform(df)
        preds = self.model.predict_proba_new(features, **kwargs)

        if self.data_transformer.target_transform == 'sparse_encoding':
            df_probas = pd.DataFrame(
                preds, index=df.index, columns=self.data_transformer.target_encoder.classes_[-kwargs['n_label']:]
            )
            return df_probas.fillna(0)

        return pd.DataFrame(preds, index=df.index, columns=self.data_transformer.target_encoder.classes_)

    def predict_score(self, df, **kwargs):
        """
        Predict probabilities over target space for feature in df.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------

        """
        features = self.data_transformer.transform(df)
        preds = self.model.predict_score(features, **kwargs)

        if self.data_transformer.target_transform == 'sparse_encoding':
            df_probas = pd.DataFrame(
                preds, index=df.index, columns=self.data_transformer.target_encoder.classes_[-kwargs['n_label']:]
            )
            return df_probas

        return pd.DataFrame(preds, index=df.index, columns=self.data_transformer.target_encoder.classes_)

    def evaluate(self, df_train, df_test):
        """
        Compute different metrics for evaluation of the current classifier performance using labelled data set.

        Parameters
        ----------
        df_train : pandas.DataFrame
            Corpus of labelled document to evaluate the classifier.

        df_test : pandas.DataFrame
            Corpus of labelled document to evaluate the classifier.

        Returns
        -------

        """

        # Build features
        X_train, y_train = self.data_transformer.transform(df_train, target=True)
        X_test, y_test = self.data_transformer.transform(df_test, target=True)

        # Transform target for scoring 1D array of class indices
        y_train, y_test = transform_label_for_scoring(y_train), transform_label_for_scoring(y_test)

        # Instantiate model and fit it
        yhat_train = self.model.predict(X_train)
        yhat_test = self.model.predict(X_test)

        # Compute confusion matrix
        confmat_train = confusion_matrix(yhat_train, y_train)
        confmat_test = confusion_matrix(yhat_test, y_test)

        # Compute multiple score
        d_scores = {'train': compute_scores(yhat_train, y_train, self.data_transformer.target_encoder)}
        d_scores.update({'test': compute_scores(yhat_test, y_test, self.data_transformer.target_encoder)})

        return confmat_train, confmat_test, d_scores


def get_model(model_name, model_params):
    """

    Parameters
    ----------
    model_name
    model_params

    Returns
    -------

    """

    if model_name == 'rf':
        return RandomForestClassifier(**model_params), {}

    elif model_name == 'dt':
        return DecisionTreeClassifier(**model_params), {}

    elif model_name == 'xgb':
        return xgboost.XGBClassifier(**model_params), {}

    elif model_name == 'yala':
        return Yala(**model_params), {}

    else:
        raise ValueError('Model name not understood: {}'.format(model_name))


def get_score(scoring, yhat, y, average='macro', labels=None):

    y = transform_label_for_scoring(y)

    if scoring == 'precision':
        if labels is None and average is not None:
            labels = list(set(y).intersection(set(yhat)))

        score = precision_score(y, yhat, labels=labels, average=average)

    elif scoring == 'accuracy':
        score = accuracy_score(y, yhat)

    elif scoring == 'balanced_accuracy':
        score = balanced_accuracy_score(y, yhat)

    elif scoring == 'f1_score':
        if labels is None and average is not None:
            labels = list(set(y).intersection(set(yhat)))

        score = f1_score(y, yhat, labels=labels, average=average)

    elif scoring == 'roc':
        # One hot encode the targets
        enc = OneHotEncoder(categories='auto').fit(y[:, np.newaxis])

        # Compute score
        score = roc_auc_score(
            enc.transform(y[:, np.newaxis]).toarray(), enc.transform(yhat[:, np.newaxis]).toarray(), average=average
        )

    else:
        raise ValueError('Scoring name not understood: {}'.format(scoring))

    return score


def compute_scores(yhat, y, label_encoder):

    # Init score dict
    d_scores = {}

    # Compute precision for each predicted label
    l_predicted_labels = list(set(yhat).intersection(y))
    l_precisions = get_score('precision', yhat, y, average=None, labels=l_predicted_labels)
    d_scores['precision'] = {label_encoder.classes_[l_predicted_labels[i]]: p for i, p in enumerate(l_precisions)}
    d_scores['label_unpredicted'] = [label_encoder.classes_[i] for i in set(y).difference(yhat)]

    # Compute other global metrics
    d_scores['accuracy'] = get_score('accuracy', yhat, y)
    d_scores['balanced_accuracy'] = get_score('balanced_accuracy', yhat, y)
    d_scores['f1_score'] = get_score('f1_score', yhat, y, average='micro')

    # Compute roc_auc for each predicted label
    l_rocs = get_score('roc', yhat, y, average=None)
    d_scores['roc_auc'] = {label_encoder.classes_[i]: p for i, p in enumerate(l_rocs)}

    return d_scores


def transform_label_for_scoring(y):

    if isinstance(y, spmatrix):
        if y.shape[1] == 1:
            y = y.toarray()[:, 0].astype(int)

        else:
            y = y.toarray().argmax(axis=1)

    return y
