import os, tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import labsonar_ml.model.auto_encoder as ml_ae
import app.config as config


trainings = [config.Training.AE_REAL, config.Training.AE_SYNTHETIC]
source_synthetics = [config.Training.GAN]

batch_size=32
n_epochs=64
lr=1e-3
latent_dim=16
steps=2

reset=False
backup=True
train = True
evaluate = True
one_fold_only = True

skip_folds = []

ml_utils.print_available_device()
config.make_dirs()
config.set_seed()

class_list = ['A','B','C','D']

for training in trainings:
    if reset and train:
        ml_utils.prepare_train_dir(config.get_result_dir(0, training), backup=backup)
        config.make_dirs()

for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), desc="Folds", leave=False):

    if i_fold in skip_folds:
        continue

    for training in trainings:

        if training == config.Training.AE_REAL:

            ae_dir = config.get_result_dir(i_fold, training, config.Artifacts.MODEL)
            ae_output_dir = config.get_result_dir(i_fold, training, config.Artifacts.OUTPUT)

            if train:
                for id, class_id in enumerate(class_list):

                    ae_file = os.path.join(ae_dir, f'{class_id}_model.plk')
                    ae_loss = os.path.join(ae_dir, f'{class_id}_loss_history.png')
                    if os.path.exists(ae_file):
                        continue

                    image = train_dataset.__getitem__(0)[0]
                    image_dim = list(image.shape)
                    ae = ml_ae.AE(feature_dim=image_dim[1],
                                  latent_dim=latent_dim,
                                  steps=steps)
                    
                    errors = ae.fit(dataset=train_dataset.filt_dataset(class_id),
                           n_epochs=n_epochs,
                           batch_size=batch_size,
                           lr=lr,
                           device=ml_utils.get_available_device())

                    ae.save(ae_file)

                    batchs = range(1, errors.shape[0] + 1)
                    plt.figure(figsize=(10, 6))
                    plt.plot(batchs, errors, label='Error')
                    plt.xlabel('Batchs')
                    plt.ylabel('Error')
                    plt.yscale('log')
                    plt.title(f'Classifier {class_id} from real data')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(ae_loss)
                    plt.close()

            if evaluate:

                for id, class_id in enumerate(class_list):

                    ae_file = os.path.join(ae_dir, f'{class_id}_model.plk')
                    if os.path.exists(ae_file):
                        continue

                    ae = ml_model.Serializable(ae_file)

                    reconstructed, loss = ae.reconstruction(dataset=val_dataset.filt_dataset(class_id),
                           device=ml_utils.get_available_device())

                    


    if one_fold_only:
        break


    # for class_id in train_dataset.get_classes():

    #     model_dir = config.get_result_dir(i_fold, training_dict['dir'], config.Artifacts.MODEL)
    #     output_dir = config.get_result_dir(i_fold, training_dict['dir'], config.Artifacts.OUTPUT)
    #     output_dir = os.path.join(output_dir, class_id)
    #     os.makedirs(output_dir, exist_ok=True)

    #     trainer_file = os.path.join(model_dir, f'{class_id}_model.plk')
    #     training_loss_file = os.path.join(model_dir, f'{class_id}_loss_history.png')
    #     training_sample_mp4 = os.path.join(model_dir, f'{class_id}_sample.mp4')

    #     if train:

    #         if os.path.exists(trainer_file):
    #             continue

    #         # train.set_specialist_class(class_id)
    #         class_train_dataset = train_dataset.filt_dataset(class_id)

    #         trainer = ml_gan.Gan_trainer(type = training_dict['type'],
    #                                     latent_space_dim = training_dict['latent_space_dim'],
    #                                     n_epochs = training_dict['n_epochs'],
    #                                     lr = training_dict['lr'])
    #         errors = trainer.fit(data = class_train_dataset, export_progress_file=training_sample_mp4)

    #         trainer.save(trainer_file)

    #         batchs = range(1, errors.shape[0] + 1)
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(batchs, errors[:,0], label='Discriminator Real Error')
    #         plt.plot(batchs, errors[:,1], label='Discriminator Fake Error')
    #         plt.plot(batchs, errors[:,2], label='Generator Error')
    #         plt.xlabel('Batchs')
    #         plt.ylabel('Error')
    #         plt.title('Generator and Discriminator Errors per Epoch')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.savefig(training_loss_file)
    #         plt.close()

    #     if evaluate:

    #         if not os.path.exists(trainer_file):
    #             continue

    #         trainer = ml_model.Serializable.load(trainer_file)
    #         images = trainer.generate_images(n_samples=training_dict['n_samples'])

    #         for index, image in enumerate(images):
    #             image_file = os.path.join(output_dir, f'{index}.png')
    #             image.save(image_file)





