import pandas as pd
import numpy as np
import tensorflow.keras as K
import mlflow.tensorflow
import sys
import logging


# mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000


if __name__ == '__main__':

    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)
    # mlflow
    mlflow.tensorflow.autolog()

    # args
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    train_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    test_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    Y_cols = ['Survived']

    data = pd.read_csv('../data/titanic/train.csv', usecols=train_cols)
    data['Sex_b'] = pd.factorize(data.Sex)[0]
    data = data.drop(['Sex'], axis=1)
    data = data.rename(columns={"Sex_b": "Sex"})

    test = pd.read_csv('../data/titanic/test.csv', usecols=test_cols)
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

    model = K.models.Sequential()
    model.add(K.layers.Dense(units=512, input_dim=6,
                             kernel_initializer='ones',
                             kernel_regularizer=K.regularizers.l1(l1_ratio),
                             ))
    model.add(K.layers.Dense(units=1, activation='sigmoid'))

    opt = K.optimizers.Adam(learning_rate=alpha)

    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'],)
    history = model.fit(x=X_train,
                        y=Y_train,
                        batch_size=128,
                        epochs=50,
                        validation_split=0.2)
    score = model.predict(test_input_data, batch_size=32, verbose=1)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])
