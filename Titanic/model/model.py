import pandas as pd
import numpy as np
import tensorflow.keras as K
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    Y_cols = ['Survived']

    data = pd.read_csv('../data/titanic/train.csv', usecols=cols)
    data['Sex_b'] = pd.factorize(data.Sex)[0]
    data = data.drop(['Sex'], axis=1)
    data = data.rename(columns={"Sex_b": "Sex"})

    # filling na values with mean
    column_means = data.mean()
    data = data.fillna(column_means)

    input_data = np.array(data[X_cols])
    label_date = np.array(data[Y_cols])

    X_train, X_val, Y_train, Y_val = train_test_split(input_data,
                                                      label_date,
                                                      test_size=0.33,
                                                      random_state=42)

    # model definition
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=128, input_dim=6))
    model.add(K.layers.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=X_train,
              y=Y_train,
              batch_size=128,
              epochs=50,
              validation_data=(X_val, Y_val))
