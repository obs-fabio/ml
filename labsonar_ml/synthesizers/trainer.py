import typing
import os
import tqdm
import abc
import numpy as np
import imageio
import PIL
from abc import abstractmethod

import torch
import torch.utils.data as torch_data
import torchvision

import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils

class Base_trainer(ml_model.Serializable_model, abc.ABC):

    def __init__(self, n_epochs: int, batch_size: int) -> None:
        self.image_dim = None
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    @abstractmethod
    def train_init(self, image_dim: typing.List[float]):
        pass

    @abstractmethod
    def train_step(self, samples: torch.Tensor, i_epoch: int, n_epochs: int, i_batch: int, n_batchs: int) -> np.ndarray:
        pass

    @abstractmethod
    def generate_samples(self, n_samples):
        pass

    def generate_images(self, n_samples, transform = None) -> PIL.Image:

        if transform is None:
            transform = torchvision.transforms.Normalize(mean= -1, std= 2)

        generated_samples = self.generate_samples(n_samples)

        desnorm_imgs = transform(generated_samples)
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
            export_progress_file: str = None,
            max_output_file_seconds: int = 15) -> np.ndarray:

        data_loader = torch_data.DataLoader(data, batch_size=self.batch_size, shuffle=True)
        n_batchs = len(data_loader)
        image = data.__getitem__(0)[0]
        self.image_dim = list(image.shape)

        self.train_init(self.image_dim)

        self.error_list = []
        training_images = []

        if export_progress_file is not None:
            video_target_size = max_output_file_seconds * 30 # 30fps
            if self.n_epochs > video_target_size:
                interval = self.n_epochs // video_target_size
            else:
                interval = 1

        for i_epoch in tqdm.tqdm(range(self.n_epochs), leave=False, desc="Epochs"):

            acum_error = []
            for i_batch, (samples, _) in enumerate(data_loader):
                error = self.train_step(samples, i_epoch, self.n_epochs, i_batch, n_batchs)
                acum_error.append(list(error))

            self.error_list.append(np.mean(acum_error, axis=0))

            if export_progress_file is not None:
                if i_epoch % interval == 0:
                    training_images.append(self.generate_images(1)[0])

        if export_progress_file is not None:
            _, extension = os.path.splitext(export_progress_file)

            if extension == ".gif":
                imageio.mimsave(export_progress_file, training_images, 'GIF', duration=1/30)
            else:
                imageio.mimsave(export_progress_file, training_images, format='FFMPEG', codec='libx264', fps=30)

        return np.array(self.error_list)

