import typing
import numpy as np
import enum
from overrides import overrides

import torch
import torchvision

import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.synthesizers.trainer as ml_trainer
# import labsonar_ml.model.unet as ml_unet
import labsonar_ml.synthesizers.diffusion_model.unet as ml_unet
# import labml_nn.diffusion.ddpm.unet as ml_unet

class Sampling_strategy(enum.Enum):
    DDPM = 1
    DDPI = 2


class DiffusionModel(ml_trainer.Base_trainer):

    def __init__(self,
                 betas = torch.linspace(1e-4, 2e-2, 501, device=ml_utils.get_available_device()),
                 n_epochs: int = 32,
                 batch_size: int = 32,
                 lr: int = 1e-3):
        super().__init__(n_epochs = n_epochs, batch_size = batch_size)

        self.betas = betas
        self.timesteps = len(betas) - 1
        self.lr = lr

        self.alphas = 1 - self.betas
        self.acum_alphas = torch.cumsum(self.alphas.log(), dim=0).exp() # verificar se exp(cumsum(log)) é realmente numéricamente mais estavel q cumprod
        self.acum_alphas[0] = 1

        self.device = ml_utils.get_available_device()
        self.sampling_strategy = Sampling_strategy.DDPM
        self.ddpi_steps_factor = self.timesteps/4

    def add_step_noise(self, samples):
        noise = torch.randn_like(samples)
        time_step = torch.randint(1, self.timesteps + 1, (samples.shape[0], )).to(self.device)
        noised_samples = self.acum_alphas.sqrt()[time_step, None, None, None] * samples + (1 - self.acum_alphas[time_step, None, None, None]) * noise
        return noised_samples, time_step, noise

    def denoise_ddpm_sampling(self, samples, timestep, predicted_noise):
        # Remover o ruído predito e adicionar rúido de volta para evitar que colapse o treinamento
        z = torch.randn_like(samples) if timestep != 1 else 0
        noise = self.betas.sqrt()[timestep] * z
        mean = (samples - predicted_noise *
                    ((1 - self.alphas[timestep]) / (1 - self.acum_alphas[timestep]).sqrt())
                ) / self.alphas[timestep].sqrt()
        return mean + noise

    def denoise_ddim_sampling(self, samples, timestep, prev_timestep, predicted_noise):
        acum_alpha = self.acum_alphas[timestep]
        prev_acum_alpha = self.acum_alphas[prev_timestep]

        x_0_pred = acum_alpha.sqrt() * ((samples - (1 - acum_alpha).sqrt() * predicted_noise) / acum_alpha.sqrt())
        dir_x_t = (1 - acum_alpha).sqrt() * predicted_noise

        return x_0_pred + dir_x_t

    @overrides
    def train_init(self, image_dim: typing.List[float]):
        self.image_dim = image_dim
        # self.model = ml_unet.UNet(image_channels=image_dim[0], n_channels=image_dim[0]**2).to(self.device)
        self.model = ml_unet.ContextUnet(in_channels = 1,
                                 n_feat = 5 * (28 **2),
                                 n_cfeat = 1,
                                 height = 28).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)


    @overrides
    def train_step(self, samples) -> np.ndarray:
        self.optimizer.zero_grad()

        samples = samples.to(self.device)

        noised_samples, timesteps, noise = self.add_step_noise(samples)
        timesteps = timesteps / self.timesteps

        timesteps = timesteps.to(self.device)
        predicted_noise = self.model(noised_samples, timesteps)

        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        loss.backward()
        self.optimizer.step()

        return np.array([loss.tolist()])

    @torch.no_grad()
    def generate_samples(self, n_samples):

        if self.model is None:
            raise UnboundLocalError('Generating imagens for not fitted models')

        self.model.eval()
        samples = torch.randn(n_samples, *self.image_dim).to(self.device)

        step_size = self.timesteps // self.ddpi_steps_factor if self.sampling_strategy == Sampling_strategy.DDPI else 1

        for timestep in range(self.timesteps, 0, -step_size):

            normalized_timestep = torch.tensor([timestep / self.timesteps])[:, None, None, None].to(self.device)

            if self.sampling_strategy == Sampling_strategy.DDPM:

                predicted_noise = self.model(samples, normalized_timestep)
                samples = self.denoise_ddpm_sampling(samples, timestep, predicted_noise)

            elif self.sampling_strategy == Sampling_strategy.DDPI:
                predicted_noise = self.model(samples, normalized_timestep)
                samples = self.denoise_ddim_sampling(samples, timestep, timestep -step_size, predicted_noise)
            else:
                raise UnboundLocalError("Sampling strategy not implemented")

        return samples


