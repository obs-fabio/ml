import functools
import torch

import labsonar_ml.model.base_model as ml_model

class MLP(ml_model.Base):
    def __init__(self, input_shape, n_neurons, dropout=0.2):
        super().__init__()
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.model = None

        image_dim = functools.reduce(lambda x, y: x * y, input_shape[1:])

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(1),
            torch.nn.Linear(image_dim, self.n_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.n_neurons, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        if self.model is None:
            self.start(x.shape)

        y = self.model(x)
        return y