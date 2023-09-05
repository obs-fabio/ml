import os, tqdm
import numpy as np
import torch
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import app.config as config

trainings_dict = [
    {
        'type': ml_gan.Type.GAN,
        'dir': config.Training.GAN,
        'batch_size': 32,
        'n_epochs': 1024,
        'latent_space_dim': 128,
        'n_samples': 256,
        'lr': 2e-4,
        'gen_cycles': 1,
        'n_bins': 0,
        'alternate_training': False,
        'mod_chance': 1,
        'lr_factor': 0.8,
    },
    # {
    #     'type': ml_gan.Type.GAN_BIN,
    #     'dir': config.Training.GANSPE,
    #     'batch_size': 32,
    #     'n_epochs': 10000,
    #     'latent_space_dim': 128,
    #     'n_samples': 256,
    #     'lr': 2e-4,
    #     'gen_cycles': 1,
    #     'n_bins': 0,
    #     'alternate_training': False,
    #     'mod_chance': 1,
    #     'lr_factor': 0.8,
    # }
]
# 'alternate_training': False, 'mod_chance': 1, 'lr_factor': 1,
# 'alternate_training': False, 'mod_chance': 1, 'lr_factor': 0.8,   ----  X
# 'alternate_training': False, 'mod_chance': 0.8, 'lr_factor': 0.8,
# 'alternate_training': False, 'mod_chance': 1, 'lr_factor': 0.6,
# 'alternate_training': True, 'mod_chance': 0.1, 'lr_factor': 1,
# 'alternate_training': True, 'mod_chance': 0.05, 'lr_factor': 0.6,
# 'alternate_training': False, 'mod_chance': 1, 'lr_factor': 0.7,
# 'alternate_training': False, 'mod_chance': 0.7, 'lr_factor': 0.8,
# 'alternate_training': False, 'mod_chance': 0.5, 'lr_factor': 0.6,
# 'alternate_training': False, 'mod_chance': 1, 'lr_factor': 0.8,   ----  X
# 'alternate_training': False, 'mod_chance': 1, 'lr_factor': 0.9,
# 'alternate_training': False, 'mod_chance': 1, 'lr_factor': 0.7,   ----  X


# 'alternate_training': False, 'mod_chance': 1, 'lr_factor': 0.8,   'latent_space_dim': 256,
# 'alternate_training': False, 'mod_chance': 1, 'lr_factor': 0.8,   'latent_space_dim': 64,


selections = {
	'A': [range(2,9), range(15,23), range(31,39), range(71,99)],
	'B': [range(2,8), range(26,32), range(38,44), range(65,91)],
	'C': [range(1,8), range(14,19), range(28,35), range(70,75)],
	'D': [range(1,10), range(14,18), range(33,38), range(55,62)],
}


bin_selections = {}
for id, list_index in selections.items():
    bin_selections[id] = []
    for list in list_index:
        for i in list:
            bin_selections[id].append(i)


reset=True
backup=True
train = True
evaluate = True
one_fold_only = True
one_class_only = True

skip_folds = []
skip_class = []

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

        for class_id in train_dataset.get_classes():

            if class_id in skip_class:
                continue

            model_dir = config.get_result_dir(i_fold, training_dict['dir'], config.Artifacts.MODEL)
            output_dir = config.get_result_dir(i_fold, training_dict['dir'], config.Artifacts.OUTPUT)
            output_dir = os.path.join(output_dir, class_id)
            os.makedirs(output_dir, exist_ok=True)

            trainer_file = os.path.join(model_dir, f'{class_id}_model.plk')
            training_gen_loss_file = os.path.join(model_dir, f'{class_id}_gen_loss_history.png')
            training_disc_loss_file = os.path.join(model_dir, f'{class_id}_disc_loss_history.png')
            training_disc_spe_loss_file = os.path.join(model_dir, f'{class_id}_disc_spe_loss_history.png')
            training_sample_mp4 = os.path.join(model_dir, f'{class_id}_sample.mp4')

            if train:

                if os.path.exists(trainer_file):
                    continue

                # train.set_specialist_class(class_id)
                class_train_dataset = train_dataset.filt_dataset(class_id)

                selected_bins = bin_selections[class_id]

                trainer = ml_gan.Gan_trainer(type = training_dict['type'],
                                            latent_space_dim = training_dict['latent_space_dim'],
                                            n_epochs = training_dict['n_epochs'],
                                            lr = training_dict['lr'],
                                            n_g = training_dict['gen_cycles'],
                                            bins=selected_bins,
                                            alternate_training = training_dict['alternate_training'],
                                            mod_chance = training_dict['mod_chance'],
                                            lr_factor = training_dict['lr_factor'])

                errors = trainer.fit(data = class_train_dataset, export_progress_file=training_sample_mp4)

                print(errors[-1,:])

                trainer.save(trainer_file)

                epochs = range(1, errors.shape[0] + 1)

                plt.figure(figsize=(10, 6))
                plt.plot(epochs, errors[:,2], label='Generator Error')
                plt.xlabel('epochs')
                plt.ylabel('Error')
                plt.title('Generator and Discriminator Errors per Batch')
                plt.grid(True)
                plt.savefig(training_gen_loss_file)
                plt.close()

                plt.figure(figsize=(10, 6))
                plt.plot(epochs, errors[:,0], label='Discriminator Real Error')
                plt.plot(epochs, errors[:,1], label='Discriminator Fake Error')
                plt.xlabel('epochs')
                plt.ylabel('Error')
                plt.title('Generator and Discriminator Errors per Epoch')
                plt.legend()
                plt.grid(True)
                plt.savefig(training_disc_loss_file)
                plt.close()

                if errors.shape[1] >= 5:
                    plt.figure(figsize=(10, 6))
                    plt.plot(epochs, errors[:,3], label='Discriminator(Bin) Real Error')
                    plt.plot(epochs, errors[:,4], label='Discriminator(Bin) Fake Error')
                    plt.xlabel('epochs')
                    plt.ylabel('Error')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(training_disc_spe_loss_file)
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