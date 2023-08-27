import torch
from abc import ABC, abstractmethod
from overrides import overrides

import labsonar_ml.model.base_model as ml_model

class Generator(ml_model.Base, ABC):

    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def make_noise(self, n_samples: int, device):
        pass

    def generate(self, n_samples: int, device):
        return self(self.make_noise(n_samples=n_samples, device=device))


class GAN(Generator):
    
    def __init__(self, latent_dim, feature_dim, internal_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.internal_dim = internal_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, internal_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(internal_dim, internal_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(internal_dim * 2, feature_dim),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

    @overrides
    def make_noise(self, n_samples: int, device):
        return torch.autograd.variable.Variable(torch.randn(n_samples, self.latent_dim)).to(device)
        # return torch.randn(n_samples, self.latent_dim, device=device)


class DCGAN(Generator):

    def __init__(self, n_channels: int, latent_dim: int, feature_dim: int):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(self.latent_dim, self.feature_dim * 4, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(self.feature_dim * 4),
            torch.nn.ReLU(True),
            # state size - (feature_dim*4) x 4 x 4
            torch.nn.ConvTranspose2d(self.feature_dim * 4, self.feature_dim * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.feature_dim * 2),
            torch.nn.ReLU(True),
            # state size - (feature_dim*2) x 8 x 8
            torch.nn.ConvTranspose2d(self.feature_dim * 2, self.feature_dim, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.feature_dim),
            torch.nn.ReLU(True),
            # state size - (feature_dim) x 16 x 16
            torch.nn.ConvTranspose2d(self.feature_dim, self.n_channels, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size - n_channels x feature_dim x feature_dim
        )

    def forward(self, input):
        return self.main(input)

    @overrides
    def make_noise(self, n_samples: int, device):
        return torch.autograd.variable.Variable(torch.randn(n_samples, self.latent_dim, 1, 1)).to(device)
        # return torch.randn(n_samples, self.latent_dim, 1, 1, device=device).clone().detach()
