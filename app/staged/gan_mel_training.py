import os, tqdm
import numpy as np
import torch
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.gan.trainer as ml_gan
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import app.config as config

trainings_dict = [
    {
        'type': ml_gan.Type.GAN,
        'dir': config.Training.GAN,
        'batch_size': 32,
        'n_epochs': 2048,
        'latent_space_dim': 128,
        'n_samples': 256,
        'lr': 2e-4,
        'gen_cycles': 1,
        'n_bins': 0
    },
    {
        'type': ml_gan.Type.GAN_BIN,
        'dir': config.Training.GANBIN_10,
        'batch_size': 32,
        'n_epochs': 2048,
        'latent_space_dim': 128,
        'n_samples': 256,
        'lr': 2e-4,
        'gen_cycles': 1,
        'n_bins': 10
    },
    {
        'type': ml_gan.Type.GAN_BIN,
        'dir': config.Training.GANBIN_20,
        'batch_size': 32,
        'n_epochs': 2048,
        'latent_space_dim': 128,
        'n_samples': 256,
        'lr': 2e-4,
        'gen_cycles': 1,
        'n_bins': 20
    },
    {
        'type': ml_gan.Type.GAN_BIN,
        'dir': config.Training.GANBIN_30,
        'batch_size': 32,
        'n_epochs': 2048,
        'latent_space_dim': 128,
        'n_samples': 256,
        'lr': 2e-4,
        'gen_cycles': 1,
        'n_bins': 30
    },
    {
        'type': ml_gan.Type.GAN_BIN_Y,
        'dir': config.Training.GANBIN_Y,
        'batch_size': 32,
        'n_epochs': 2048,
        'latent_space_dim': 128,
        'n_samples': 256,
        'lr': 2e-4,
        'gen_cycles': 1,
        'n_bins': 0
    },
    # {
    #     'type': ml_gan.Type.DCGAN,
    #     'dir': config.Training.DCGAN,
    #     'batch_size': 8,
    #     'n_epochs': 2048,
    #     'latent_space_dim': 128,
    #     'lr': 2e-4,
    #     'gen_cycles': 5,
    #     'n_samples': 256,
    # }
]


reset=False
backup=False
train = True
evaluate = True
one_fold_only = False
one_class_only = False

skip_folds = []

ml_utils.print_available_device()
config.make_dirs()

for training_dict in tqdm.tqdm(trainings_dict, desc="Tipos"):
    if reset and train:
        ml_utils.prepare_train_dir(config.get_result_dir(0, training_dict['dir']), backup=backup)
        config.make_dirs()
            

for training_dict in tqdm.tqdm(trainings_dict, desc="Tipos"):
    for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), desc=f"{training_dict['type'].name.lower()}_Fold", leave=False):

        if i_fold in skip_folds:
            continue

        bin_dir = config.get_result_dir(i_fold, config.Training.BASELINE, config.Artifacts.OUTPUT)
        bin_files = os.path.join(bin_dir, f"baseline_f1.txt")
        bin_indexes = np.loadtxt(bin_files, dtype=int)

        for class_id in train_dataset.get_classes():

            model_dir = config.get_result_dir(i_fold, training_dict['dir'], config.Artifacts.MODEL)
            output_dir = config.get_result_dir(i_fold, training_dict['dir'], config.Artifacts.OUTPUT)
            output_dir = os.path.join(output_dir, class_id)
            os.makedirs(output_dir, exist_ok=True)

            trainer_file = os.path.join(model_dir, f'{class_id}_model.plk')
            training_loss_file = os.path.join(model_dir, f'{class_id}_loss_history.png')
            training_sample_mp4 = os.path.join(model_dir, f'{class_id}_sample.mp4')

            if train:

                if os.path.exists(trainer_file):
                    continue

                # train.set_specialist_class(class_id)
                class_train_dataset = train_dataset.filt_dataset(class_id)

                if training_dict['n_bins'] == 0:
                    selected_bins = None
                else:
                    selected_bins = bin_indexes[:training_dict['n_bins']]

                trainer = ml_gan.Gan_trainer(type = training_dict['type'],
                                            latent_space_dim = training_dict['latent_space_dim'],
                                            n_epochs = training_dict['n_epochs'],
                                            lr = training_dict['lr'],
                                            n_g = training_dict['gen_cycles'],
                                            bins=selected_bins)
                errors = trainer.fit(data = class_train_dataset, export_progress_file=training_sample_mp4)

                trainer.save(trainer_file)

                batchs = range(1, errors.shape[0] + 1)
                plt.figure(figsize=(10, 6))
                plt.plot(batchs, errors[:,0], label='Discriminator Real Error')
                plt.plot(batchs, errors[:,1], label='Discriminator Fake Error')
                plt.plot(batchs, errors[:,2], label='Generator Error')
                if errors.shape[1] >= 5:
                    plt.plot(batchs, errors[:,3], label='Discriminator(Bin) Real Error')
                    plt.plot(batchs, errors[:,4], label='Discriminator(Bin) Fake Error')
                plt.xlabel('Batchs')
                plt.ylabel('Error')
                plt.yscale('log')
                plt.title('Generator and Discriminator Errors per Epoch')
                plt.legend()
                plt.grid(True)
                plt.savefig(training_loss_file)
                plt.close()

            if evaluate:

                if not os.path.exists(trainer_file):
                    continue

                trainer = ml_model.Serializable.load(trainer_file)
                images = trainer.generate_images(n_samples=training_dict['n_samples'])

                for index, image in enumerate(images):
                    image_file = os.path.join(output_dir, f'{index}.png')
                    image.save(image_file)

            if one_class_only:
                break

        if one_fold_only:
            break