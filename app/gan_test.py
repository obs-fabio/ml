import os, tqdm
import numpy as np
import torch
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils

data_dir = '/tf/ml/data/'
base_dir = '/tf/ml/test_results/gan'
output_dir = 'output'
training_dir = 'training'
batch_size = 32
latent_space_dim=100
n_epochs=20
n_samples=100
lr = 2e-4
reset=False
backup_old = False
train = True
evalueate = True

def get_mnist_dataset_as_specialist(datapath: str = "data/", batch_size: int = 32, specialist_class_number: int = 1):
    transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(32, antialias=True),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.5,), (0.5,))])
    
    train = torchvision.datasets.MNIST(root=datapath, train=True, download=True, transform=transform)
    
    targets_idx = train.targets.detach().clone() == specialist_class_number

    return torch_data.dataset.Subset(train, np.where(targets_idx)[0])

output_dir = os.path.join(base_dir, output_dir)
training_dir = os.path.join(base_dir, training_dir)

if train:

    if reset:
        ml_utils.prepare_train_dir(base_dir, backup=backup_old)
    else:
        os.makedirs(base_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)

    for class_id in tqdm.tqdm(range(1), desc="Class"):

        trainer_file = os.path.join(training_dir, 'trainer_{:d}.plk'.format(class_id))
        training_history_file = os.path.join(training_dir, 'training_history_{:d}.png'.format(class_id))

        if os.path.exists(trainer_file) and \
            os.path.exists(training_history_file):
            continue

        train = get_mnist_dataset_as_specialist(datapath = data_dir, batch_size = batch_size, specialist_class_number = class_id)

        trainer = ml_gan.Gan_trainer(latent_space_dim = latent_space_dim,
                                     n_epochs = n_epochs,
                                     lr = lr)
        errors = trainer.fit(data = train, export_progress_file=os.path.join(training_dir, "training_history_{:d}.gif".format(class_id)))

        trainer.save(trainer_file)
        epochs = range(1, errors.shape[0] + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, errors[:,0], label='Generator Error')
        plt.plot(epochs, errors[:,1], label='Discriminator Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Generator and Discriminator Errors per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(training_history_file)
        plt.close()

if evalueate:

    for class_id in tqdm.tqdm(range(1), desc="Class"):

        trainer_file = os.path.join(training_dir, 'trainer_{:d}.plk'.format(class_id))
        
        class_output_dir = os.path.join(output_dir,"{:d}".format(class_id))
        os.makedirs(class_output_dir, exist_ok=True)

        if not os.path.exists(trainer_file):
            continue

        trainer = ml_model.Serializable.load(trainer_file)
        images = trainer.generate(n_samples=n_samples)

        for index, image in enumerate(images):
            image_file = os.path.join(class_output_dir, '{:d}.png'.format(index))
            image.save(image_file)


