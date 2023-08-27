import typing
import os
import tqdm
import abc
from abc import abstractmethod
import numpy as np
import imageio

import torch.utils.data as torch_data

import labsonar_ml.model.base_model as ml_model

class Base_trainer(ml_model.Serializable, abc.ABC):

    def __init__(self, n_epochs: int, batch_size: int) -> None:
        self.image_dim = None
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    @abstractmethod
    def train_init(self, image_dim: typing.List[float]):
        pass

    @abstractmethod
    def train_step(self, samples) -> np.ndarray:
        """realiza um step do treinamento

        Args:
            samples (_type_): tensor batch_size x imagem

        Returns:
            np.ndarray: matrix de erros do treinamento, (modelos,) - considerado como somatorio dos erros do batch
        """
        pass

    @abstractmethod
    def generate(self, n_samples, transform = None):
        pass

    def fit(self,
            data: torch_data.Dataset,
            export_progress_file: str = None) -> np.ndarray:

        data_loader = torch_data.DataLoader(data, batch_size=self.batch_size, shuffle=True)
        image = data.__getitem__(0)[0]
        self.image_dim = list(image.shape)

        self.train_init(self.image_dim)

        self.error_list = []
        training_images = []
        for epoch in tqdm.tqdm(range(self.n_epochs), leave=False, desc="Epochs"):
            epoch_error_accum = None

            for n_batch, (samples, _) in enumerate(data_loader):
                error = self.train_step(samples)

                if epoch_error_accum is None:
                    epoch_error_accum = error
                else:
                    error = np.sum([epoch_error_accum, error])

                if export_progress_file is not None:
                    training_images.append(self.generate(1)[0])

            self.error_list.append(list(epoch_error_accum/len(data_loader.dataset)))


        if export_progress_file is not None:
            imageio.mimsave(export_progress_file, training_images, 'GIF', duration=0.5)

        return np.array(self.error_list)
