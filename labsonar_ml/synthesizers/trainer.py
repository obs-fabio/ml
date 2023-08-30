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

        video_target_size = 15*30

        self.error_list = []
        training_images = []
        for _ in tqdm.tqdm(range(self.n_epochs), leave=False, desc="Epochs"):

            for bacth, (samples, _) in enumerate(data_loader):
                error = self.train_step(samples)
                self.error_list.append(list(error))

                if export_progress_file is not None and self.n_epochs < video_target_size:
                    training_images.append(self.generate(1)[0])

            if export_progress_file is not None and self.n_epochs >= video_target_size:
                training_images.append(self.generate(1)[0])

        if export_progress_file is not None:
            if len(training_images) > video_target_size:
                n_images = len(training_images)
                interval = n_images // video_target_size
                remainder = n_images % video_target_size
                
                filtered_images = []
                for i in range(video_target_size):
                    idx = i * interval + min(i, remainder)
                    filtered_images.append(training_images[idx])
                training_images = filtered_images

            imageio.mimsave(export_progress_file, training_images, format='FFMPEG', codec='libx264', fps=30)
            # imageio.mimsave(export_progress_file, training_images, 'GIF', duration=1/60)

        return np.array(self.error_list)
