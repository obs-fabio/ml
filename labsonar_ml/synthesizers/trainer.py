import typing
import os
import tqdm
import abc
from abc import abstractmethod
import numpy as np
import imageio
import PIL

import torch.utils.data as torch_data
import torchvision

import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils

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
    def generate_samples(self, n_samples):
        pass

    def generate_images(self, n_samples, transform = None):

        if transform is None:
            transform = torchvision.transforms.Normalize(mean= -1, std= 2)

        generated_samples = self.generate_samples(n_samples)
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

    def fit(self,
            data: torch_data.Dataset,
            export_progress_file: str = None) -> np.ndarray:

        data_loader = torch_data.DataLoader(data, batch_size=self.batch_size, shuffle=True)
        image = data.__getitem__(0)[0]
        self.image_dim = list(image.shape)

        self.train_init(self.image_dim)

        self.error_list = []
        training_images = []
        for _ in tqdm.tqdm(range(self.n_epochs), leave=False, desc="Epochs"):

            for bacth, (samples, _) in enumerate(data_loader):
                error = self.train_step(samples)
                self.error_list.append(list(error))

            if export_progress_file is not None:
                training_images.append(self.generate_images(1)[0])

        if export_progress_file is not None:
            imageio.mimsave(export_progress_file, training_images, 'GIF', duration=5/len(training_images)) # salva giff com duração de 5segundos independente do numero de batch de treinamento

        return np.array(self.error_list)

