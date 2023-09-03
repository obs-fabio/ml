import math
import torch

import labsonar_ml.model.base_model as ml_model

class GAN(ml_model.Base):
    def __init__(self, feature_dim, internal_dim=256, dropout=0.2):
        super().__init__()
        self.internal_dim = internal_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, internal_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(internal_dim * 2, internal_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
        )
        self.reset_output_layer()

    def reset_output_layer(self):
        self.activation = torch.nn.Sequential(
            torch.nn.Linear(self.internal_dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        output = self.activation(output)
        return output


class DCGAN(ml_model.Base):
    def __init__(self, n_channels: int, feature_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.n_channels = n_channels
        self.feature_dim = feature_dim

        final_layer_size = 8

        num_layers = int(round(math.log2(feature_dim)-math.log2(final_layer_size)-1)) # reduzir feature_dim/2 -> 4 - considerando seguidas divisões por 2

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

        # state size - (batch_size) x (final_layer_size x final_layer_size)
        layers.extend([
            torch.nn.Conv2d(self.feature_dim * (2**num_layers), 1, 4, 1, 0, bias=False),
        ])

        # state size - (batch_size) x (1 x 1)
        self.model = torch.nn.Sequential(*layers)

        self.mlp = torch.nn.Sequential(
            torch.nn.Flatten(1),
            torch.nn.Linear((final_layer_size-3)**2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(self.model(x))


# class DCGAN(ml_model.Base):
#     def __init__(self, n_channels: int, feature_dim: int, negative_slope: float = 0.2):
#         super().__init__()
#         self.n_channels = n_channels
#         self.feature_dim = feature_dim

#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1), torch.nn.LeakyReLU(0.2, inplace=True), torch.nn.Dropout2d(0.25)]
#             if bn:
#                 block.append(torch.nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.model = torch.nn.Sequential(
#             *discriminator_block(n_channels, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         # The height and width of downsampled image
#         ds_size = feature_dim // 2 ** 4
#         self.adv_layer = torch.nn.Sequential(torch.nn.Linear(128 * ds_size ** 2, 1), torch.nn.Sigmoid())

#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.shape[0], -1)
#         validity = self.adv_layer(out)
#         return validity
