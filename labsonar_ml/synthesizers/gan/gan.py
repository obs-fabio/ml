from abc import ABC, abstractmethod
from overrides import overrides
import typing
import functools
import torch

import labsonar_ml.model.base_model as ml_model

class Generator(ml_model.Base, ABC):
    """ Network to convert noise from latent dimension to data dimension"""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim: int = latent_dim

    def make_noise(self, n_samples: int, device: typing.Union[str, torch.device]) -> torch.Tensor:
        return torch.autograd.variable.Variable(torch.randn(n_samples, self.latent_dim)).to(device)

    def generate(self, n_samples: int, device: typing.Union[str, torch.device]) -> torch.Tensor:
        return self(self.make_noise(n_samples=n_samples, device=device))

class Generator_MLP(Generator):

    def __init__(self, latent_dim: int, data_dims: typing.List[int], internal_dims: typing.List[int] = [256, 512]):
        super().__init__(latent_dim)
        self.data_dims: typing.List[int] = data_dims
        self.data_size = functools.reduce(lambda x, y: x * y, data_dims)
        self.internal_dims = internal_dims

        layer = [
            torch.nn.Linear(latent_dim, self.internal_dims[0]),
            torch.nn.LeakyReLU(),
        ]

        for i in range(1,len(self.internal_dims)):
            layer.extend([
                torch.nn.Linear(self.internal_dims[i-1], self.internal_dims[i]),
                torch.nn.LeakyReLU(),
            ])

        layer.extend([
            torch.nn.Linear(self.internal_dims[-1], self.data_size),
            torch.nn.Tanh(),
        ])

        self.model = torch.nn.Sequential(*layer)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        x = self.model(x)
        return x.view(n_samples, *self.data_dims)



class Discriminator(ml_model.Base, ABC):
    """ Network to classify the data dimension in real (true) or false (false)"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset_activation_layer(self):
        """ reset last layer to transfer learning """
        pass

class Discriminator_MLP(Discriminator):
    def __init__(self, data_dims: typing.List[int], internal_dims: typing.List[int] = [512, 256], dropout: float = 0.2):
        super().__init__()
        self.internal_dims = internal_dims
        self.data_size = functools.reduce(lambda x, y: x * y, data_dims)

        layer = [
            torch.nn.Flatten(1),
            torch.nn.Linear(self.data_size, self.internal_dims[0]),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        ]
        for i in range(1,len(self.internal_dims)):
            layer.extend([
                torch.nn.Linear(self.internal_dims[i-1], self.internal_dims[i]),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout)
            ])

        self.model = torch.nn.Sequential(*layer)
        self.reset_activation_layer()

    @overrides
    def reset_activation_layer(self):
        self.activation = torch.nn.Sequential(
            torch.nn.Linear(self.internal_dims[-1], 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.activation(x)
        return x

