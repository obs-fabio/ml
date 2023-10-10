import functools
import typing
import torch

import labsonar_ml.model.base_model as ml_model

class MLP(ml_model.Base):
    def __init__(self,
                 input_shape: typing.List[int],
                 n_neurons: int,
                 dropout: float = 0.2,
                 activation_hidden_layer: torch.nn.Module = torch.nn.LeakyReLU(),
                 activation_output_layer: torch.nn.Module = torch.nn.Sigmoid()):
        super().__init__()
        self.n_neurons = n_neurons
        self.model = None

        image_dim = functools.reduce(lambda x, y: x * y, input_shape[1:])

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(1),
            torch.nn.Linear(image_dim, self.n_neurons),
            activation_hidden_layer,
            torch.nn.Dropout(dropout),
        )

        self.activation = torch.nn.Sequential(
            torch.nn.Linear(self.n_neurons, 1),
            activation_output_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.activation(x)
        return x