import keras as K
load_yaml = __import__('utils').load_yalm


def create_model():
    _, model_params = load_yaml()

    network = model_params.get('layers')
    l1_ratio = model_params.get('l1')
    alpha = model_params.get('lr')
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
