import torch

import labsonar_ml.model.base_model as ml_model

class Generator(ml_model.Base):
    def __init__(self, input_dim, output_dim, internal_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.internal_dim = internal_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, internal_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(internal_dim, internal_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(internal_dim * 2, output_dim),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

    def make_noise(self, n_samples: int, device):
        return torch.randn(n_samples, self.input_dim, device=device).clone().detach()

    def generate(self, n_samples: int, device):
        return self(self.make_noise(n_samples=n_samples, device=device)).detach().clone()