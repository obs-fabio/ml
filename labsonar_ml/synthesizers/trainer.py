import typing
import os
import tqdm
import abc
from abc import abstractmethod
import numpy as np
import imageio
import PIL
import random

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
    def train_step(self, samples, rand) -> np.ndarray:
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
        for i in tqdm.tqdm(range(n_samples), leave=False):
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
        
        video_target_size = 15*30

        self.error_list = []
        training_images = []
        for i_epochs in tqdm.tqdm(range(self.n_epochs), leave=False, desc="Epochs"):

            rand = random.uniform(0, 1)
            acum_error = []
            for batch, (samples, _) in enumerate(data_loader):
                error = self.train_step(samples, rand)
                # self.error_list.append(list(error))
                acum_error.append(list(error))
                
                if export_progress_file is not None and self.n_epochs < video_target_size:
                    training_images.append(self.generate_images(1)[0])

            self.error_list.append(np.mean(acum_error, axis=0))

            if export_progress_file is not None and self.n_epochs >= video_target_size:
                training_images.append(self.generate_images(1)[0])

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

