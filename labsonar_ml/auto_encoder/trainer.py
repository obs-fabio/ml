import tqdm
import typing
import numpy as np
import abc
from abc import abstractmethod

import torch
import torch.utils.data as torch_data

import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.synthesizers.trainer as ml_train
import labsonar_ml.auto_encoder.model as ml_ae

class AE_trainer(ml_model.Serializable, abc.ABC):

    def __init__(self, n_epochs: int, batch_size: int, lr: float, latent_dim: int, steps: int) -> None:
        self.image_dim = None
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.latent_dim = latent_dim
        self.steps = steps
        self.device = ml_utils.get_available_device()

    @abstractmethod
    def train_init(self, image_dim: typing.List[float]):
        self.model = ml_ae.AE(feature_dim=image_dim[1],
                        latent_dim=self.latent_dim,
                        steps=self.steps)

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
        for i in tqdm.tqdm(range(n_samples), leave=False):
            data = (desnorm_imgs[i].permute(1, 2, 0)).numpy()
            data = data.reshape((data.shape[0], data.shape[1]))
            data = (data * 255).astype(np.uint8)
            images.append(PIL.Image.fromarray(data, mode='L'))

        return images

    def fit(self, dataset: torch_data.Dataset) -> np.ndarray:

        data_loader = torch_data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss()

        self.to(self.device)


        image = dataset.__getitem__(0)[0]
        image_dim = list(image.shape)
        self.train_init(image_dim)

        self.error_list = []
        for _ in tqdm.tqdm(range(self.n_epochs), leave=False, desc="Epochs"):

            for bacth, (samples, _) in enumerate(data_loader):

                optimizer.zero_grad()

                samples = samples.to(self.device)
                reconstructed = self(samples)

                loss = loss_func(reconstructed, ml_utils.images_to_vectors(samples))
                loss.backward()
                optimizer.step()

                self.error_list.append(loss.item())

        return np.array(self.error_list)


