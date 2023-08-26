import numpy as np
import PIL
from overrides import overrides

import torch
import torchvision

import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.synthesizers.gan.generator as gan_g
import labsonar_ml.synthesizers.gan.discriminator as gan_d
import labsonar_ml.synthesizers.trainer as ml_train


class Gan_trainer(ml_train.Base_trainer):

    def __init__(self,
                 latent_space_dim: int,
                 g_model: gan_g.Generator = None, 
                 d_model: gan_d.Discriminator = None,
                 g_optimizador = None,
                 d_optimizador = None,
                 loss_func = None,
                 lr: float = 2e-4,
                 n_epochs: int = 100,
                 batch_size: int = 32):
        super().__init__(n_epochs, batch_size)

        self.g_model = g_model
        self.d_model = d_model
        self.g_optimizador = g_optimizador
        self.d_optimizador = d_optimizador
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

        return (real_loss + fake_loss).tolist()

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
    def train_init(self, data_size: float):
        
        self.g_model = self.g_model if self.g_model is not None else \
                gan_g.Generator(input_dim = self.latent_space_dim,
                                output_dim = data_size)
        self.d_model = self.d_model if self.d_model is not None else \
                gan_d.Discriminator(input_dim = data_size)
        
        self.g_model = self.g_model.to(self.device)
        self.d_model = self.d_model.to(self.device)

        self.g_optimizador = self.g_optimizador if self.g_optimizador is not None else \
                torch.optim.Adam(self.g_model.parameters(), lr = self.lr)
        self.d_optimizador = self.d_optimizador if self.d_optimizador is not None else \
                torch.optim.Adam(self.d_model.parameters(), lr = self.lr)

    @overrides
    def train_step(self, samples) -> np.ndarray:
        n_samples = samples.size(0)

        real_data = torch.autograd.variable.Variable(ml_utils.images_to_vectors(samples))
        fake_data = self.g_model.generate(n_samples, self.device)
        d_error = self.discriminator_step(real_data, fake_data)

        fake_data = self.g_model.generate(n_samples, self.device)
        g_error = self.generator_step(fake_data)

        return np.array([d_error, g_error])
    
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
