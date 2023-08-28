import os, tqdm
import torch.utils.data as torch_data

import labsonar_ml.data_loader as ml_data
import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan

batch_size = 32
latent_space_dim=100
n_epochs=100







import os, tqdm
import numpy as np
import torch
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils

# types = [ml_gan.Type.GAN]
# types = [ml_gan.Type.DCGAN]
types = [ml_gan.Type.GAN, ml_gan.Type.DCGAN]

data_dir = '/tf/ml/data/4classes'
base_dir = '/tf/ml/results/'
output_dir = 'output'
training_dir = 'training'
batch_size = 32
latent_space_dim=256
n_epochs=2000
n_samples=128
lr = 2e-4
reset=False
backup_old = True
train = False
evaluate = True
one_run_only = False

print(ml_utils.print_available_device())

for type in types:

    print("Iniciando o treinamento da ", type.name.lower())

    type_dir = os.path.join(base_dir, type.name.lower())

    output_dir = os.path.join(type_dir, output_dir)
    training_dir = os.path.join(type_dir, training_dir)

    custom_dataset = ml_data.init_four_classes_dataset(data_dir)

    if train:

        if reset:
            ml_utils.prepare_train_dir(type_dir, backup=backup_old)
        else:
            os.makedirs(type_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(training_dir, exist_ok=True)

        for class_id, loro in tqdm.tqdm(custom_dataset.get_specialist_loro()):

            for run_index, (train, test) in tqdm.tqdm(enumerate(loro), leave=False):

                trainer_file = os.path.join(training_dir, 'trainer_class({:s})_{:d}.plk'.format(class_id, run_index))
                training_history_file = os.path.join(training_dir, 'training_history_class({:s})_{:d}.png'.format(class_id, run_index))

                if os.path.exists(trainer_file) and \
                    os.path.exists(training_history_file):
                    continue

                trainer = ml_gan.Gan_trainer(type = type,
                                            latent_space_dim = latent_space_dim,
                                            n_epochs = n_epochs,
                                            lr = lr)
                errors = trainer.fit(data = train, export_progress_file=os.path.join(training_dir, "training_history_class({:s})_{:d}.gif".format(class_id, run_index)))

                trainer.save(trainer_file)
                epochs = range(1, errors.shape[0] + 1)

                plt.figure(figsize=(10, 6))
                plt.plot(epochs, errors[:,0], label='Discriminator Real Error')
                plt.plot(epochs, errors[:,1], label='Discriminator Fake Error')
                plt.plot(epochs, errors[:,2], label='Generator Error')
                plt.xlabel('Epochs')
                plt.ylabel('Error')
                plt.title('Generator and Discriminator Errors per Epoch')
                plt.legend()
                plt.grid(True)
                plt.savefig(training_history_file)
                plt.close()
                
                if one_run_only:
                    break

            if one_run_only:
                break

    if evaluate:

        for class_id, loro in tqdm.tqdm(custom_dataset.get_specialist_loro()):

            for run_index, (train, test) in tqdm.tqdm(enumerate(loro), leave=False):

                trainer_file = os.path.join(training_dir, 'trainer_class({:s})_{:d}.plk'.format(class_id, run_index))

                if not os.path.exists(trainer_file):
                    continue

                class_output_dir = os.path.join(output_dir,"{:s}".format(class_id))
                os.makedirs(class_output_dir, exist_ok=True)

                trainer = ml_model.Serializable.load(trainer_file)
                images = trainer.generate(n_samples=n_samples)

                for index, image in enumerate(images):
                    image_file = os.path.join(class_output_dir, '{:d}_{:d}.png'.format(run_index, index))
                    image.save(image_file)

                if one_run_only:
                    break

            if one_run_only:
                break


