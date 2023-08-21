import labsonar_ml.models.base as ml
from labsonar_ml.models.mlp import MLP

import numpy as np
import pandas as pd

from tensorflow import keras

class Voting_ensemble(ml.Base):
    def __init__(self, descriptions, **kwargs):
        super().__init__()
        self.descriptions = descriptions

    def fit(self, X, Y, **kwargs):
        return

    def predict(self, X,  **kwargs):
        predict = np.zeros((X.shape[0], len(self.descriptions)))
        for i, description in enumerate(self.descriptions):
            model = ml.Base.load(description['filename'])
            prediction = (model.predict(X) > description['decision_threshold']).astype(int)
            predict[:, i] = prediction.flatten()


        sum_vector = np.sum(predict, axis=1, keepdims=True)
        ret = (sum_vector > len(self.descriptions)/2.0).astype(int)
        return ret
    

class MLP_ensemble(ml.Base):
    def __init__(self, descriptions, **kwargs):
        super().__init__()
        self.descriptions = descriptions
        self.model = MLP(n_hidden=4, regularize=0.005, batch_size=1/6, best_model="./results/temp/aux.h5", **kwargs)

    def transform_input(self, X):
        transform = np.zeros((X.shape[0], len(self.descriptions)))
        for i, description in enumerate(self.descriptions):
            model = ml.Base.load(description['filename'])
            prediction = model.predict(X)
            transform[:, i] = prediction.flatten()
        return pd.DataFrame(transform)

    def fit(self, X, Y, **kwargs):
        val_X = kwargs.get('val_X')
        val_Y = kwargs.get('val_Y')

        XT = self.transform_input(X)
        val_XT = self.transform_input(val_X)

        self.model.fit(XT, Y, val_X=val_XT, val_Y=val_Y)

    def predict(self, X, **kwargs):
        XT = self.transform_input(X)
        return self.model.predict(XT)

