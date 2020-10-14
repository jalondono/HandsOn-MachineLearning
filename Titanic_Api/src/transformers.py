from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self._feature_names = feature_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        return X[self._feature_names]


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """Transformer class"""

    def __init__(self, feature_names):
        """Constructor method"""
        self._feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """ Preprocess the data """
        return X[self._feature_names]


class NumericalTransformer(BaseEstimator, TransformerMixin):
    # # Class Constructor
    # def __init__(self, feature_names):
    #     self._feature_names = feature_names

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    # Custom transform method
    def transform(self, X, y=None):
        return X


class fill_nans(BaseEstimator, TransformerMixin):
    # # Class Constructor
    # def __init__(self, feature_names):
    #     self._feature_names = feature_names

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    # Custom transform method
    def transform(self, X, y=None):
        # zero to nan
        # X['Fare'] = X['Fare'].replace(0, np.nan)

        # fill Fare
        X['Fare'] = X.groupby(['Pclass', 'Sex'])['Fare'] \
            .transform(lambda x: x.fillna(x.mean()))

        # fill Age
        X['Age'] = X.groupby(['Pclass', 'Sex'])['Age'] \
            .transform(lambda x: x.fillna(x.mean()))

        # print('data set shape: ', X.shape, '\n')
        # print(X.head())
        # print(X.columns[data.isna().any()].tolist())
        # print('**********Aqui*************')
        return X
