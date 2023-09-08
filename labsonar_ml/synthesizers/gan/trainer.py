from overrides import overrides
import numpy as np
import typing

import torch

import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.synthesizers.gan.gan as ml_gan
import labsonar_ml.synthesizers.trainer as ml_train


class GAN_trainer(ml_train.Base_trainer):

    def __init__(self,
                 n_epochs: int = 100,
                 batch_size: int = 32,
                 # generator properties
                 latent_space_dim: int = 128,
                 g_lr: float = 2e-4,
                 g_n_cycles: int = 1,
                 g_internal_dims: typing.List[int] = [256, 512],
                 # discriminator properties
                 d_lr: float = 2e-4,
                 d_n_cycles: int = 1,
                 d_internal_dims: typing.List[int] = [512, 256],
                 d_dropout: float = 0.2,
                 ):
        super().__init__(n_epochs, batch_size)

        self.latent_space_dim: int = latent_space_dim
        self.g_lr: float = g_lr
        self.g_n_cycles: int = g_n_cycles
        self.g_internal_dims: typing.List[int] = g_internal_dims
        self.d_lr: float = d_lr
        self.d_n_cycles: int = d_n_cycles
        self.d_internal_dims: typing.List[int] = d_internal_dims
        self.d_dropout: float = d_dropout

        self.device = ml_utils.get_available_device()
        self.loss_func = torch.nn.BCELoss()

    def discriminator_step(self, real_data: torch.Tensor) -> typing.List[float]:
        """ Realiza um passo de treinamento no discriminador
        Returns:
            typing.List[float]: acurária do discriminador para amostras reais e falsas
        """
        n_samples = real_data.size(0)
        fake_data = self.generate_samples(n_samples)

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

        real_erros = np.sum(real_pred.detach().cpu().numpy()>0.5)
        fake_erros = np.sum(fake_pred.detach().cpu().numpy()<0.5)

        return [real_erros/n_samples, fake_erros/n_samples]

    def generator_step(self, real_data: torch.Tensor):
        n_samples = real_data.size(0)
        fake_data = self.generate_samples(n_samples)

        self.g_optimizador.zero_grad()

        pred = self.d_model(fake_data)
        loss = self.loss_func(pred, ml_utils.make_targets(n_samples, 1, self.device))
        loss.backward()

        self.g_optimizador.step()

    @overrides
    def train_init(self, image_dim: typing.List[float]):

        self.g_model = ml_gan.Generator_MLP(latent_dim=self.latent_space_dim,
                                            data_dims=image_dim,
                                            internal_dims=self.g_internal_dims)
        self.d_model = ml_gan.Discriminator_MLP(data_dims = image_dim,
                                                internal_dims = self.d_internal_dims,
                                                dropout = self.d_dropout)

        self.g_model = self.g_model.to(self.device)
        self.d_model = self.d_model.to(self.device)
        self.g_optimizador = torch.optim.Adam(self.g_model.parameters(), lr = self.g_lr)
        self.d_optimizador = torch.optim.Adam(self.d_model.parameters(), lr = self.d_lr)

    @overrides
    def train_step(self, samples: torch.Tensor, i_epoch: int, n_epochs: int, i_batch: int, n_batchs: int) -> np.ndarray:
        
        # print(i_batch, "/", n_batchs, " de ", i_epoch, "/",  n_epochs)

        for _ in range(self.d_n_cycles):
            d_erros = self.discriminator_step(samples)

        for _ in range(self.g_n_cycles):
            self.generator_step(samples)

        return np.array(d_erros)

    def generate_samples(self, n_samples) -> torch.Tensor:
        if self.g_model is None:
            raise UnboundLocalError('Generating imagens for not fitted models')
        return self.g_model.generate(n_samples=n_samples, device=self.device)


class SPECGAN_trainer(GAN_trainer):

    def __init__(self,
                 n_epochs: int = 100,
                 batch_size: int = 32,
                 # generator properties
                 latent_space_dim: int = 128,
                 g_lr: float = 2e-4,
                 g_n_cycles: int = 1,
                 g_internal_dims: typing.List[int] = [256, 512],
                 # discriminator properties
                 d_lr: float = 2e-4,
                 d_n_cycles: int = 1,
                 d_internal_dims: typing.List[int] = [512, 256],
                 d_dropout: float =0.2,
                 # specialist discriminator properties
                 sd_lr: float = 2e-4,
                 sd_internal_dims: typing.List[int] = [512, 256],
                 sd_dropout: float =0.2,
                 sd_bins: typing.List[int] = [],
                 sd_reg_factor: float = 1,
                 ):
        super().__init__(n_epochs,batch_size,
                 latent_space_dim,g_lr,g_n_cycles,g_internal_dims,
                 d_lr,d_n_cycles,d_internal_dims,d_dropout,)

        self.sd_lr = sd_lr
        self.sd_internal_dims = sd_internal_dims
        self.sd_dropout = sd_dropout
        self.sd_bins = [*sd_bins] # nao sei pq sem isso a lista vira uma tupla e nada funciona
        self.sd_reg_factor = sd_reg_factor

    @overrides
    def discriminator_step(self, real_data: torch.Tensor) -> typing.List[float]:
        """ Realiza um passo de treinamento no discriminador
        Returns:
            typing.Tuple[float, float]: acurária do discriminador para amostras reais e falsas, acurária do discriminador especialista para amostras reais e falsas
        """
        d_erros = super().discriminator_step(real_data)
        n_samples = real_data.size(0)
        fake_data = self.generate_samples(n_samples)

        self.sd_optimizador.zero_grad()

        real_data = real_data[:, :, self.sd_bins, :]
        fake_data = fake_data[:, :, self.sd_bins, :]

        real_data = real_data.to(self.device)
        fake_data = fake_data.to(self.device)

        real_pred = self.sd_model(real_data)
        real_loss = self.loss_func(real_pred, ml_utils.make_targets(n_samples, 1, self.device))
        real_loss.backward(retain_graph=True)

        fake_pred = self.sd_model(fake_data)
        fake_loss = self.loss_func(fake_pred, ml_utils.make_targets(n_samples, 0, self.device))
        fake_loss.backward(retain_graph=True)

        self.sd_optimizador.step()

        real_erros = np.sum(real_pred.detach().cpu().numpy()>0.5)
        fake_erros = np.sum(fake_pred.detach().cpu().numpy()<0.5)

        return d_erros + [real_erros/n_samples, fake_erros/n_samples]

    @overrides
    def generator_step(self, real_data: torch.Tensor):
        n_samples = real_data.size(0)
        fake_data = self.generate_samples(n_samples)

        self.g_optimizador.zero_grad()

        d_pred = self.d_model(fake_data)
        d_loss = self.loss_func(d_pred, ml_utils.make_targets(n_samples, 1, self.device))
        d_loss.backward(retain_graph=True)

        fake_data = fake_data[:, :, self.sd_bins, :]

        sd_pred = self.sd_model(fake_data)
        sd_loss = self.loss_func(sd_pred, ml_utils.make_targets(n_samples, 1, self.device)) * self.sd_reg_factor
        sd_loss.backward(retain_graph=True)

        self.g_optimizador.step()

    @overrides
    def train_init(self, image_dim: typing.List[float]):
        super().train_init(image_dim)

        selected_image_dim = image_dim.copy()
        selected_image_dim[1] = len(self.sd_bins)

        self.sd_model = ml_gan.Discriminator_MLP(data_dims = selected_image_dim,
                                                internal_dims = self.sd_internal_dims,
                                                dropout = self.sd_dropout)
        
        self.sd_model = self.sd_model.to(self.device)
        self.sd_optimizador = torch.optim.Adam(self.sd_model.parameters(), lr = self.sd_lr)
