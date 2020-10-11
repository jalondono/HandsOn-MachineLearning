import numpy as np
import pandas as pd
import warnings
import typer
import yaml
import logging
import zipfile
import mlflow.sklearn
from urllib.parse import urlparse

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from keras_model import create_model

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


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
        clf = KerasClassifier(build_fn=create_model, verbose=1,
                              epochs=args['epochs'],
                              data={'data': args})
        # ****************************************************
        return clf
    if model_name == "LR":
        return LinearRegression()

    if model_name == "TD":
        return DecisionTreeClassifier(random_state=42)


def load_hyperparams():
    """
    load the hyper params from a yml file
    :param args: dictionary with the data
    :return: a dictionary with hyper params
    """
    with open("parameters.yml", "r") as f:
        params = yaml.safe_load(f)
    return params


app = typer.Typer()


@app.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def main(
        ctx: typer.Context,
        model_type: str = "NN",
        data_path: str = "data/titanic.zip",
        debug: bool = False,
        toy: bool = False
):
    data_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    cat_cols = ['Pclass', 'Sex']
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare']

    data = load_data(data_path, filename='train.csv', cols=data_cols)
    X = data.drop('Survived', axis=1)
    # You can covert the target variable to numpy
    y = data['Survived'].values
    with mlflow.start_run():
        full_pipeline = build_pipeline(num_cols, cat_cols)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ********************
        # load the model
        params = load_hyperparams()
        model_params = params['NN']
        model = choose_model(model_type, model_params)

        full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline),
                                          ('model', model)])

        # Can call fit on it just like any other pipeline
        trained = full_pipeline_m.fit(X_train, y_train)

        # Can predict with it like any other pipeline
        y_pred = full_pipeline_m.predict(X_test)
        n_corr = np.sum(y_pred[:, 0] == y_test)
        print(f'Correct percentage: {n_corr / len(y_test)}')
        # print accuracy
        # accuracy= \
        #     round(trained.score(X_test, y_test) * 100, 2)
        # print(acc_decision_tree_test)

        # ********************************************************
        # mlflow log params
        (rmse, mae, r2) = eval_metrics(y_test, y_pred[:, 0])
        mlflow.log_param("alpha", 0.005)
        mlflow.log_param("l1_ratio", 0)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(0.005, "model", registered_model_name="Neural_Network")
        else:
            mlflow.sklearn.log_model(0.005, "model")


if __name__ == "__main__":
    app()
