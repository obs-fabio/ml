import math
import torch

import labsonar_ml.model.base_model as ml_model

class GAN(ml_model.Base):
    def __init__(self, feature_dim, internal_dim=256, dropout=0.2):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, internal_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(internal_dim * 2, internal_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(internal_dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class DCGAN(ml_model.Base):
    def __init__(self, n_channels: int, feature_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.n_channels = n_channels
        self.feature_dim = feature_dim

        num_layers = int(round(math.log2(feature_dim)-3)) # reduzir feature_dim/2 -> 4 - considerando seguidas divisões por 2

        # input is batch_size x (n_channels x feature_dim x feature_dim)  - batch x imagem
        layers = [
            torch.nn.Conv2d(self.n_channels, self.feature_dim, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(negative_slope, inplace=True)
        ]

        # state size - (batch_size) x (feature_dim/2 x feature_dim/2)
        for i in range(num_layers):
            layers.extend([
                torch.nn.Conv2d(self.feature_dim * (2**i), self.feature_dim * (2**(i+1)), 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(self.feature_dim * (2**(i+1))),
                torch.nn.LeakyReLU(negative_slope, inplace=True)
            ])

        # state size - (batch_size) x (4 x 4)
        layers.extend([
            torch.nn.Conv2d(self.feature_dim * (2**num_layers), 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        ])

        # state size - (batch_size) x (1 x 1)
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output.reshape(output.shape[0], output.shape[1])