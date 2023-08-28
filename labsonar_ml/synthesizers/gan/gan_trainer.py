import numpy as np
import PIL
import enum
from overrides import overrides
import functools
import typing

import torch
import torchvision

import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.synthesizers.gan.generator as gan_g
import labsonar_ml.synthesizers.gan.discriminator as gan_d
import labsonar_ml.synthesizers.trainer as ml_train

class Type(enum.Enum):
    GAN = 1,
    DCGAN = 2


class Gan_trainer(ml_train.Base_trainer):

    def __init__(self,
                 type: Type,
                 latent_space_dim: int,
                 loss_func = None,
                 lr: float = 2e-4,
                 n_epochs: int = 100,
                 batch_size: int = 32):
        super().__init__(n_epochs, batch_size)

        self.type = type
        self.loss_func = loss_func if loss_func is not None else torch.nn.BCELoss(reduction='sum')
        self.lr = lr
        self.latent_space_dim = latent_space_dim
        self.device = ml_utils.get_available_device()
        self.image_dim = None

    def discriminator_step(self, real_data, fake_data) -> float:
        n_samples = real_data.size(0)
        self.d_optimizador.zero_grad()

        real_data = real_data.to(self.device)
        fake_data = fake_data.to(self.device)

        real_pred = self.d_model(real_data)
        real_loss = self.loss_func(real_pred, ml_utils.make_targets(n_samples, 1, self.device))
        real_loss.backward()

        fake_pred = self.d_model(fake_data)
        fake_loss = self.loss_func(fake_pred, ml_utils.make_targets(n_samples, 0, self.device))
        fake_loss.backward()

        self.d_optimizador.step()

        return real_loss.tolist(), fake_loss.tolist()

    def generator_step(self, fake_data) -> float:
        n_samples = fake_data.size(0)
        self.g_optimizador.zero_grad()

        fake_data = fake_data.to(self.device)

        pred = self.d_model(fake_data)
        loss = self.loss_func(pred, ml_utils.make_targets(n_samples, 1, self.device))
        loss.backward()

        self.g_optimizador.step()

        return loss.tolist()

    @overrides
    def train_init(self, image_dim: typing.List[float]):

        if self.type == Type.GAN:

            image_dim = functools.reduce(lambda x, y: x * y, image_dim)

            self.g_model = gan_g.GAN(latent_dim = self.latent_space_dim,
                                    feature_dim = image_dim)
            self.d_model = gan_d.GAN(feature_dim = image_dim)
            
        elif self.type == Type.DCGAN:

            n_channels = image_dim[0]
            feature_dim = image_dim[1]
            
            self.g_model = gan_g.DCGAN(n_channels = n_channels,
                                        latent_dim = self.latent_space_dim,
                                        feature_dim = feature_dim)
            self.d_model = gan_d.DCGAN(n_channels = n_channels,
                                        feature_dim = feature_dim)

        else:
            raise UnboundLocalError("Model " + str(type) + " not implemented")

        self.g_model = self.g_model.to(self.device)
        self.d_model = self.d_model.to(self.device)

        self.g_optimizador = torch.optim.Adam(self.g_model.parameters(), lr = self.lr)
        self.d_optimizador = torch.optim.Adam(self.d_model.parameters(), lr = self.lr)

    @overrides
    def train_step(self, samples) -> np.ndarray:
        n_samples = samples.size(0)

        if self.type == Type.GAN:
            real_data = torch.autograd.variable.Variable(ml_utils.images_to_vectors(samples))
        else:
            real_data = samples
        fake_data = self.g_model.generate(n_samples, self.device)
        d_real_error, d_fake_error = self.discriminator_step(real_data, fake_data)

        fake_data = self.g_model.generate(n_samples, self.device)
        g_error = self.generator_step(fake_data)

        return np.array([d_real_error, d_fake_error, g_error])

    @overrides
    def generate(self, n_samples, transform = None):

        if self.g_model is None:
            raise UnboundLocalError('Generating imagens for not fitted models')
        
        if transform is None:
            transform = torchvision.transforms.Normalize(mean= -1, std= 2)

        self.g_model.eval()
        generated_samples = self.g_model.generate(n_samples=n_samples, device=self.device)
        generated_imgs = ml_utils.vectors_to_images(vectors = generated_samples, image_dim=self.image_dim)

        desnorm_imgs = transform(generated_imgs)
        desnorm_imgs = desnorm_imgs.cpu().detach()

        images = []
        for i in range(n_samples):
            data = (desnorm_imgs[i].permute(1, 2, 0)).numpy()
            data = data.reshape((data.shape[0], data.shape[1]))
            data = (data * 255).astype(np.uint8)
            images.append(PIL.Image.fromarray(data, mode='L'))

        return images
