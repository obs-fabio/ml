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
        self.model = torch.nn.Sequential(
            # input is (n_channels) x feature_dim x feature_dim
            torch.nn.Conv2d(self.n_channels, self.feature_dim, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(negative_slope, inplace=True),
            # state size - (feature_dim) x 16 x 16
            torch.nn.Conv2d(self.feature_dim, self.feature_dim * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.feature_dim * 2),
            torch.nn.LeakyReLU(negative_slope, inplace=True),
            # state size - (feature_dim*2) x 8 x 8
            torch.nn.Conv2d(self.feature_dim * 2, self.feature_dim * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.feature_dim * 4),
            torch.nn.LeakyReLU(negative_slope, inplace=True),
            # state size - (feature_dim*4) x 4 x 4
            torch.nn.Conv2d(self.feature_dim * 4, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output.reshape(output.shape[0], output.shape[1])