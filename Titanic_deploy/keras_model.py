import tensorflow.keras as K


def create_model(data):
    network = [256, 128]
    l1_ratio = 0
    alpha = 0.005
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=network[0], input_dim=9,
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
