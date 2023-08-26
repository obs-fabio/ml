import torch

import labsonar_ml.model.base_model as ml_model

class Discriminator(ml_model.Base):
    def __init__(self, input_dim, internal_dim=256, dropout=0.2):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, internal_dim * 2),
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