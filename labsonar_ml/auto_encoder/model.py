import tqdm
import numpy as np
import torch
import torch.utils.data as torch_data

import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils

class AE(ml_model.Base):
    def __init__(self, feature_dim: int, latent_dim: int, steps: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.steps = steps
        

        encoder_layers = [
            torch.nn.Flatten(1),
            torch.nn.Linear(feature_dim ** 2, latent_dim * (2**steps)),
        ]

        for i in range(steps, 0, -1):
            encoder_layers.extend([
                torch.nn.ReLU(),
                torch.nn.Linear(latent_dim * (2**i), latent_dim * (2**(i-1))),
            ])

        self.encoder = torch.nn.Sequential(*encoder_layers)


        decoder_layers = []

        for i in range(steps):
            decoder_layers.extend([
                torch.nn.Linear(latent_dim * (2**i), latent_dim * (2**(i+1))),
                torch.nn.ReLU(),
            ])

        decoder_layers.extend([
            torch.nn.Linear(latent_dim * (2**steps), feature_dim ** 2),
            torch.nn.Sigmoid()
        ])

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    @torch.no_grad
    def reconstruction(self, dataset, device):

        data_loader = torch_data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        self.to(device)

        for _, (samples, classes) in enumerate(data_loader):

            samples = samples.to(device)
            reconstructed = self(samples)

            loss_func = torch.nn.MSELoss()
            loss = loss_func(reconstructed, ml_utils.images_to_vectors(samples))
            return reconstructed, loss
