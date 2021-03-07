# Global imports
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from pandas import concat

# Local import
from datalab.dataops.utils.encoder import HybridEncoder, CatEncoder


class FoldManager(object):
    """
    FoldManager manage segmentation of data for cross validation.

    """
    allowed_methods = ['standard', 'stratified']

    def __init__(
            self, df_data, nb_folds, df_weights=None, method='standard', test_size=0.1,
            val_size=0.1, target_name='target'
    ):

        # Get base parameters
        self.target_name = target_name

        # Split data set into a train / test and Validation if necessary
        self.df_train, self.df_test = train_test_split(df_data, test_size=test_size, shuffle=True)
        self.df_val = None

        # Set weights
        self.df_weights = None
        if df_weights is not None:
            self.df_weights = df_weights

        # Set method to transform data
        self.data_transformer = None

        # Set sampling method for Kfold
        if nb_folds <= 2:
            self.kf = None
            if val_size > 0:
                self.df_train, self.df_val = train_test_split(df_data, test_size=val_size, shuffle=True)

        elif method == 'standard':
            self.kf = KFold(n_splits=nb_folds, shuffle=True)

        elif method == 'stratified':
            self.kf = StratifiedKFold(n_splits=nb_folds)

        else:
            raise ValueError('Choose Kfold method from {}'.format(FoldManager.allowed_methods))

    def reset(self):
        self.data_transformer = None

    def get_all_train_data(self, param_transform, force_recompute=False):
        """
        Build a data set composed of models. The target is also return, if specified.

        Parameters
        ----------
        param_transform : dict
            kw parameters to build features.

        force_recompute : bool
            If True, it fit feature builder with train data

        Returns
        -------
        dict

        """
        # concat train and validation if any
        if self.df_val is not None:
            df_train = concat([self.df_train, self.df_val])
        else:
            df_train = self.df_train

        # Create models builder if necessary
        if self.data_transformer is None or force_recompute:
            self.data_transformer = DataTransformer(**param_transform).build(df_train)

        X, y = self.data_transformer.transform(df_train, target=True)
        d_train = {"X": X, "y": y}

        if self.df_weights is not None:
            return {**d_train, "w": self.df_weights.loc[self.df_train.index].values}

        return d_train

    def get_train_data(self, param_transform, force_recompute=False):
        """
        Build a data set composed of models. The target is also return, if specified.

        Parameters
        ----------
        param_transform : dict
            kw parameters to build features.

        force_recompute : bool
            If True, it fit feature builder with train data

        Returns
        -------
        dict

        """
        # Create models builder if necessary
        if self.data_transformer is None or force_recompute:
            self.data_transformer = DataTransformer(**param_transform).build(self.df_train)

        X, y = self.data_transformer.transform(self.df_train, target=True)
        d_train = {"X": X, "y": y}

        if self.df_weights is not None:
            return {**d_train, "w": self.df_weights.loc[self.df_train.index].values}

        return d_train

    def get_eval_data(self, param_transform):
        """
        Build a data set composed of models. The target is also return, if specified.

        Parameters
        ----------
        param_transform : dict
            kw parameters to build features.

        Returns
        -------
        dict

        """
        if self.df_val is None:
            return None

        # Create models builder if necessary
        if self.data_transformer is None:
            self.data_transformer = DataTransformer(**param_transform).build(self.df_train)

        X, y = self.data_transformer.transform(self.df_val, target=True)
        d_eval = {"X": X, "y": y}

        if self.df_weights is not None:
            return {**d_eval, "w": self.df_weights.loc[self.df_val.index].values}

        return d_eval

    def get_test_data(self, param_transform=None):
        """
        Build test data set composed of models and transformed target.

        Parameters
        ----------
        param_transform : dict
            kw parameters to build features.

        Returns
        -------
        dict
            Features and target as

        """
        # Create models builder if necessary
        if self.data_transformer is None:
            self.data_transformer = DataTransformer(**param_transform).build(self.df_train)

        X, y = self.data_transformer.transform(self.df_test, target=True)
        d_test = {"X": X, "y": y}

        if self.df_weights is not None:
            return {**d_test, "w": self.df_weights.loc[self.df_test.index].values}

        return d_test

    def get_features(self, df):
        """
        Build feature.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to transform  into feature.

        Returns
        -------
        tuple
            Features as numpy.ndarray

        """
        assert self.data_transformer is not None, "Feature builder is None"
        return self.data_transformer.transform(df)

    def generate_folds(self, param_transform):
        """
        Generate train and validation data set. A data set is composed of models and target.

        Parameters
        ----------
        param_transform : dict
            kw parameters to build features.

        Returns
        -------
        tuple
            Composed of dict with Features and target, for train and validation etl.
            i.e: {'X': numpy.ndarray, 'y': numpy.ndarray}
        """

        if self.kf is not None:
            # Iterate over different folds
            for l_train, l_val in self.kf.split(self.df_train):
                index_train, index_val = list(self.df_train.index[l_train]), list(self.df_train.index[l_val])

                # Create models  builder if necessary
                self.data_transformer = DataTransformer(**param_transform).build(self.df_train.loc[index_train])

                # Get features
                X, y = self.data_transformer.transform(self.df_train.loc[index_train + index_val], target=True)

                # Build train / validation set
                X_train, y_train = X[l_train, :], y[l_train]
                X_val, y_val = X[l_val, :], y[l_val]

                if self.df_weights is not None:
                    w_train, w_val = self.df_weights.loc[index_train].values, self.df_weights.loc[index_val].values
                    yield {'X': X_train, 'y': y_train, 'w': w_train}, {'X': X_val, 'y': y_val, 'w': w_val}

                yield {'X': X_train, 'y': y_train}, {'X': X_val, 'y': y_val}

        else:
            d_train = self.get_train_data(param_transform, force_recompute=True)
            d_eval = self.get_eval_data(param_transform)

            yield d_train, d_eval


class DataTransformer(object):
    """
    The DataTransformer manage the transformation of processed data composed of job description labelled by normalized
    positions. Its transformation pipeline is composed of:
    """

    def __init__(
            self, feature_transform, cat_cols, num_cols, target_col, params, n_label, target_transform=None
    ):
        """

        Attributes
        ----------
        method : str
            Model to use to transform text data.

        """
        self.n_label, self.target_col, self.cat_cols, self.num_cols = n_label , target_col, cat_cols, num_cols
        self.feature_transform, self.target_transform = feature_transform, target_transform
        self.params, self.args = params, {}

        # Init model to None
        self.model, self.target_encoder, self.imputer = None, None, None

    def get_args(self):
        return {k: v for k, v in self.args.items()}

    def build(self, df_data=None, force_train=False):
        """
        Build models from processed data and perform a numerical transform of label. The processed data is composed,
        on one hand, of text description, that will transform to numerical vector using TF-ID. On the other hand of a
        text label that will be numerised using one hot encoding.

        Parameters
        ----------
        df_data : pandas.DataFrame
            DataFrame of processed data.
        params :  dict
            Kw params of the TF-IDF.
        force_train : bool
            whether to fit TF-IDF on the data.

        Returns
        -------
        self : Current instance of the class.
        """
        if self.model is None or force_train:

            if 'impute' in self.feature_transform:
                self.imputer = {
                    'cat_imput': SimpleImputer(**self.params.get('impute_cat', {})).fit(df_data[self.cat_cols]),
                    'num_imput': SimpleImputer(**self.params.get('impute_num', {})).fit(df_data[self.num_cols])
                }

            if "cat_encode" in self.feature_transform:
                self.model = CatEncoder(
                    handle_unknown='ignore', sparse=self.params.get('sparse', False),
                    dtype=self.params.get('dtype', bool)
                )
                self.model.fit(df_data[self.cat_cols])


            else:
                raise ValueError('Transformation not implemented: {}'.format(self.feature_transform))

            if self.target_transform in ('encoding', 'sparse_encoding'):
                self.target_encoder = LabelEncoder().fit(df_data[self.target_col])

        return self

    def transform(self, df_data, target=False):
        """
        Perform a numerical transform of the text in DataFrame df, using previously fitted TF-IDF.
        In addition, perform a numerical transform of target if specified, using a multiple class hot encoding.

        Parameters
        ----------
        df_data : pandas.DataFrame
            DataFrame of processed data that shall be transformed.

        target : bool
            Whether to transform target or not.

        Returns
        -------
        numpy.array
            Transformed data.

        """
        if 'impute' in self.feature_transform:
            X_cat = self.imputer['cat_imput'].transform(df_data[self.cat_cols])
            X_num = self.imputer['num_imput'].transform(df_data[self.num_cols])

        else:
            X_cat, X_num = df_data[self.cat_cols], df_data[self.num_cols]

        if self.feature_transform == 'cat_encode':
            X_cat = self.model.transform(X_cat)

        X = np.hstack([X_cat, X_num])

        if target:
            if self.target_transform == 'encoding':
                y = self.target_encoder.transform(df_data[self.target_col])

            elif self.target_transform == 'sparse_encoding':
                y = self.target_encoder.transform(df_data[self.target_col])

                if self.n_label > 1:
                    y = csc_matrix(([True] * y.shape[0], (range(y.shape[0]), y)), shape=(y.shape[0], len(np.unique(y))))

                else:
                    y = csc_matrix(y[:, np.newaxis] > 0)

            else:
                y = df_data.loc[:, [self.target_col]].values

            return X, y

        return X
