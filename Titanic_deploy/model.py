import numpy as np
import pandas as pd
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import typer
import logging
import zipfile


def load_data(zipfolder, filename, cols):
    """
    Get the data from a zip file
    :param path: direction to zip file
    :return: train dataset
    """
    with zipfile.ZipFile(zipfolder, 'r') as zip_ref:
        zip_ref.extractall()

    data = pd.read_csv(filename, usecols=cols)

    # print('data set shape: ', data.shape, '\n')
    # print(data.head())

    return data


# Custom Transformer that extracts columns passed as argument to its constructor
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


def build_pipeline(num_cols, cat_cols):
    """
    :param num_cols:
    :param cat_cols:
    :return:
    """
    # Numerical Pipeline
    num_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(num_cols)),
                                   ('num_transformer', NumericalTransformer()),
                                   ('imputer', SimpleImputer(strategy='median')),
                                   ('std_scaler', StandardScaler())])

    # Categorical Pipeline
    cat_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(cat_cols)),
                                   ('cat_transformer', CategoricalTransformer(cat_cols)),
                                   ('one_hot_encoder', OneHotEncoder(sparse=False))])
    # Combining numerical and categorical piepline into one full big pipeline horizontally
    # using FeatureUnion
    full_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', cat_pipeline),

                                                   ('numerical_pipeline', num_pipeline)])
    # Combining the custum imputer with the categorical and numerical pipeline
    preprocess_pipeline = Pipeline(steps=[('fill', fill_nans()),
                                          ('full_pipeline', full_pipeline)])
    return preprocess_pipeline


if __name__ == '__main__':
    data_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    cat_cols = ['Pclass', 'Sex']
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare']

    data = load_data('data/titanic.zip', filename='train.csv', cols=data_cols)
    X = data.drop('Survived', axis=1)
    # You can covert the target variable to numpy
    y = data['Survived'].values

    full_pipeline = build_pipeline(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # X_train.info()
    # # print('_' * 40)
    # X_test.info()

    full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline),
                                      ('model', LinearRegression())])

    # Can call fit on it just like any other pipeline
    trained = full_pipeline_m.fit(X_train, y_train)

    # Can predict with it like any other pipeline
    y_pred = full_pipeline_m.predict(X_test)
    print(y_pred)
    print(40*'*')
    # print accuracy
    acc_decision_tree_test = \
        round(trained.score(X_test, y_test) * 100, 2)
    print(acc_decision_tree_test)
