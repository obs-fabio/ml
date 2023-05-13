import ml.models.base as ml

from tensorflow import keras

class MLP(ml.Base):
    def __init__(self, **kwargs):
        super().__init__()
        self.n_hidden = kwargs.get('n_hidden', 10)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.random_state = kwargs.get('random_state', None)
        self.hidden_function = kwargs.get('hidden_function', 'tanh')
        self.output_function = kwargs.get('output_function', 'tanh')
        self.kernel_initializer = kwargs.get('kernel_initializer', 0.01)
        self.batch_size = kwargs.get('batch_size', 1)

    def get_loss(self, loss='cat_crossent'):
        if loss == 'cat_crossent':
            return keras.losses.CategoricalCrossentropy(from_logits=False,
                                                        label_smoothing=0.0,
                                                        axis=-1,
                                                        reduction="auto",
                                                        name=loss,)
        elif loss== 'mse':
            return keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM)
        elif loss=='bi_crossent':
            return keras.losses.BinaryCrossentropy(from_logits=False,
                                                    label_smoothing=0.0,axis=-1,
                                                    reduction="auto",
                                                    name="loss_alg")

        raise NotImplementedError("get_loss " + loss) 
    
    def get_optimizer(self, optimizer='adam', learning_rate = 0.001):
        if optimizer == 'adam':
            return keras.optimizers.Adam(learning_rate=learning_rate,
                                        beta_1=0.9,beta_2=0.999,
                                        epsilon=1e-07,amsgrad=False,
                                        name="Adam",)
        raise NotImplementedError("get_optimizer " + optimizer) 

    def fit(self, X, Y, **kwargs):

        self.n_inputs = X.shape[1]
        self.n_outputs = Y.shape[1]

        self.model = keras.models.Sequential()

        self.model.add(keras.Input(shape=(self.n_inputs,)))
        hidden_layer = keras.layers.Dense(units=self.n_hidden,
                                    activation=self.hidden_function,
                                    kernel_initializer=keras.initializers.RandomNormal(stddev=self.kernel_initializer),
                                    kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                                    bias_regularizer=keras.regularizers.L2(1e-4),
                                    bias_initializer=keras.initializers.Zeros()
                                   )

        self.model.add(hidden_layer)

        output_layer = keras.layers.Dense(units=self.n_outputs,
                                    activation=self.output_function,
                                    kernel_initializer=keras.initializers.RandomNormal(stddev=self.kernel_initializer),
                                    bias_initializer=keras.initializers.Zeros()
                                   )
        self.model.add(output_layer)

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = self.learning_rate,
                                                                  decay_steps = 100,
                                                                  decay_rate = 0.9)

        optimizer = self.get_optimizer(optimizer = 'adam', learning_rate = lr_schedule)

        loss = self.get_loss(loss = 'mse')

        cat_acc_metric = keras.metrics.CategoricalAccuracy(name = "cat_acc", dtype = None)
        acc_metric = keras.metrics.Accuracy(name = "accuracy", dtype = None)
        mse_metric = keras.metrics.MeanSquaredError(name = "mse", dtype = None)
        rmse_metric = keras.metrics.RootMeanSquaredError(name = "rmse", dtype = None)

        self.model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[cat_acc_metric,
                               acc_metric,
                               mse_metric,
                               rmse_metric])
        
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=100,
                                                verbose=2,
                                                mode='auto')
    
        val_X = kwargs.get('val_X')
        val_Y = kwargs.get('val_Y')

        self.trn_history = self.model.fit(X, Y,
                             epochs=100,
                             batch_size=round(self.batch_size * X.shape[0]),
                             callbacks=[earlyStopping], 
                             verbose=2,
                             validation_data=(val_X,
                                              val_Y),
                            )

    def predict(self, X, **kwargs):
        if self.model == None:
            raise UnboundLocalError("it is not possible to predict the data without training")

        predictions = self.model.predict(X)
        predictions = (predictions > 0.5).astype(int)
        return predictions
