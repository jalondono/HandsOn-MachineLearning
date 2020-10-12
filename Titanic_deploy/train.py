import numpy as np
import pandas as pd
import warnings
import typer
import yaml
import logging
import zipfile
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import pickle

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.wrappers.scikit_learn import KerasClassifier

create_model = __import__('keras_model').create_model
load_yalm = __import__('utils').load_yalm
load_data = __import__('utils').load_data
FeatureSelector = __import__('transformers').FeatureSelector
CategoricalTransformer = __import__('transformers').CategoricalTransformer
NumericalTransformer = __import__('transformers').NumericalTransformer
fill_nans = __import__('transformers').fill_nans

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
mlflow.sklearn.autolog()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


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


def choose_model(model_name, args):
    """
    load a model
    :param args:
    :param model_name: name of the model
    :return: a model
    """
    if model_name == "NN":
        # ********************
        # load keras model
        # wrap the model using the function you created
        clf = KerasClassifier(build_fn=create_model, **args)
        # ****************************************************
        return clf
    if model_name == "LR":
        return LinearRegression()

    if model_name == "TD":
        return DecisionTreeClassifier(random_state=42)


app = typer.Typer()


@app.command()
def main(
        ctx: typer.Context,
        model_type: str = "NN",
        data_path: str = "data/titanic.zip",
        debug: bool = False,
        toy: bool = False
):
    # *****************************************************
    # Selecting data
    data_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    cat_cols = ['Pclass', 'Sex']
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare']

    data = load_data(data_path, filename='train.csv', cols=data_cols)
    X = data.drop('Survived', axis=1)
    # You can covert the target variable to numpy
    y = data['Survived'].values
    # *****************************************************
    # **************************************************
    # pipe line definition
    full_pipeline = build_pipeline(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ********************
    # load the model
    classifier, _ = load_yalm()
    model_one = choose_model(model_type, classifier)

    full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline),
                                      ('model', model_one)])
    # **************************************************

    # Can call fit on it just like any other pipeline
    trained = full_pipeline_m.fit(X_train, y_train)

    # Can predict with it like any other pipeline
    y_pred = full_pipeline_m.predict(X_test)
    if model_type == 'NN':
        n_corr = np.sum(y_pred[:, 0] == y_test)
    else:
        n_corr = np.sum(y_pred == y_test)
    print(f'Correct percentage: {n_corr / len(y_test)}')
    # print accuracy
    # accuracy= \
    #     round(trained.score(X_test, y_test) * 100, 2)
    # print(acc_decision_tree_test)

    # Log the sklearn model and register as version 1
    # pkl_filename = "model.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(trained, file)


if __name__ == "__main__":
    app()
