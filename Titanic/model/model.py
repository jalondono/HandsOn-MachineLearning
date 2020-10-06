import pandas as pd
import numpy as np
import tensorflow.keras as K
import mlflow.tensorflow
import sys
import logging
import zipfile


# mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000
def getting_data(zipfolder, filename, cols):
    """
    Get the data from a zip file
    :param path: direction to zip file
    :return: train dataset
    """
    with zipfile.ZipFile(zipfolder, 'r') as zip_ref:
        zip_ref.extractall()

    data = pd.read_csv(filename, usecols=cols)

    print('data set shape: ', data.shape, '\n')
    print(data.head())

    return data


def process_args(argv):
    """
    convert the data arguments into the needed format
    :param argv: Parameters
    :return: converted parameters
    """
    data_path = sys.argv[1] if len(sys.argv) > 1 else '../data'
    debug = sys.argv[2].lower() if len(sys.argv) > 1 else 'false'
    model_type = sys.argv[3] if len(sys.argv) > 1 else [256, 128]
    model_type = model_type[1:-1].split(',')
    splited_network = [int(x) for x in model_type]
    alpha = float(sys.argv[4]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[5]) if len(sys.argv) > 2 else 0

    return data_path, debug, splited_network, alpha, l1_ratio


def create_model(network):
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=256, input_dim=6,
                             kernel_initializer='ones',
                             kernel_regularizer=K.regularizers.l1(l1_ratio),
                             ))
    for units in network[1:]:
        model.add(K.layers.Dense(units=units,
                                 kernel_initializer='ones',
                                 kernel_regularizer=K.regularizers.l1(l1_ratio),
                                 ))
    model.add(K.layers.Dense(units=1, activation='sigmoid'))
    opt = K.optimizers.Adam(learning_rate=alpha)

    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'], )
    print(model.summary())
    return model


def train_model(model, X_train, Y_train, batch_size=128,
                epoch=80, val_split=0.1):
    """
    Perform the training of the model
    :param model: model previously compiled
    :return: history
    """
    history = model.fit(x=X_train,
                        y=Y_train,
                        batch_size=128,
                        epochs=80,
                        validation_split=0.1)
    return history


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)
    # mlflow
    mlflow.tensorflow.autolog()

    # Utils cols from data
    train_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    test_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    Y_cols = ['Survived']

    # Get value arguments
    data_path, debug, network, alpha, l1_ratio = process_args(sys.argv)

    # train Data
    filename = 'train.csv'
    data = getting_data(data_path, filename, train_cols)
    data['Sex_b'] = pd.factorize(data.Sex)[0]
    data = data.drop(['Sex'], axis=1)
    data = data.rename(columns={"Sex_b": "Sex"})

    # testing data
    filename = 'test.csv'
    test = getting_data(data_path, filename, test_cols)
    test['Sex_b'] = pd.factorize(test.Sex)[0]
    test = test.drop(['Sex'], axis=1)
    test = test.rename(columns={"Sex_b": "Sex"})

    # filling train na values with mean
    column_means = data.mean()
    data = data.fillna(column_means)

    # filling test na values with mean
    column_means = test.mean()
    test = test.fillna(column_means)

    input_data = np.array(data[X_cols])
    label_date = np.array(data[Y_cols])
    test_input_data = np.array(test[X_cols])
    X_train = input_data
    Y_train = label_date

    # definition of the model
    model = create_model(network)
    # training model
    history = train_model(model, X_train, Y_train)
    # predicting
    score = model.predict(test_input_data, batch_size=32, verbose=1)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])
