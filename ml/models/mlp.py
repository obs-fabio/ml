import ml.models.base as ml

import numpy as np

from tensorflow import keras

class MLP(ml.Base):
    def __init__(self, **kwargs):
        super().__init__()
        self.n_hidden = kwargs.get('n_hidden', 10)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.epochs = kwargs.get('epochs', 64)
        self.random_state = kwargs.get('random_state', None)
        self.hidden_function = kwargs.get('hidden_function', 'tanh')
        self.output_function = kwargs.get('output_function', 'sigmoid')
        self.batch_size = kwargs.get('batch_size', 1)
        self.best_model = kwargs.get('best_model', None)
        self.drop_out = kwargs.get('drop_out', None)
        self.regularize = kwargs.get('regularize', None)
        self.trn_history = None

    def fit(self, X, Y, **kwargs):

        self.n_inputs = X.shape[1]

        self.model = keras.models.Sequential()
        self.model.add(keras.Input(shape=(self.n_inputs,)))

        if self.regularize is None:
            hidden_layer = keras.layers.Dense(units=self.n_hidden, activation=self.hidden_function)
        else:
            hidden_layer = keras.layers.Dense(units=self.n_hidden, activation=self.hidden_function,
                                            kernel_regularizer=keras.regularizers.l2(self.regularize))
        self.model.add(hidden_layer)

        if not self.drop_out is None:
            self.model.add(keras.layers.Dropout(self.drop_out))

        output_layer = keras.layers.Dense(units=1, activation=self.output_function)
        self.model.add(output_layer)
        
        opt = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)

        self.model.compile(loss='mse',
                    optimizer=opt,
                    metrics=[keras.metrics.Recall(name = 'recall')])

        val_X = kwargs.get('val_X')
        val_Y = kwargs.get('val_Y')

        if self.best_model :
            checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=self.best_model,
                                                            monitor='val_loss',
                                                            mode='min',
                                                            save_best_only=True)

            self.trn_history = self.model.fit(x=X,
                                y=Y,
                                epochs=self.epochs,
                                batch_size=round(self.batch_size*X.shape[0]),
                                class_weight=self.get_class_weight(Y),
                                callbacks=[checkpoint_callback],
                                verbose=0,
                                validation_data=(val_X,val_Y))
            
            keras.models.load_model(self.best_model)

        else:
            self.trn_history = self.model.fit(x=X,
                                y=Y,
                                epochs=self.epochs,
                                batch_size=round(self.batch_size*X.shape[0]),
                                class_weight=self.get_class_weight(Y),
                                verbose=0,
                                validation_data=(val_X,val_Y))


    def predict(self, X, output_as_classifier=True, **kwargs):
        if self.model == None:
            raise UnboundLocalError("it is not possible to predict the data without training")

        return self.model.predict(X,verbose=0)

