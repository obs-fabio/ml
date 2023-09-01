import os, tqdm
import numpy as np
import torch
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.diffusion_model.diffusion_trainer as ml_diff
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import app.config as config

batch_size = 8
n_epochs=25
n_samples=10#256
lr = 1e-3

reset=False
backup=True
train = True
evaluate = True
one_fold_only = False
one_class_only = False

training_dir = config.Training.DIFUSSION

skip_folds = []

ml_utils.print_available_device()
config.make_dirs()


for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), leave=False):

    if i_fold in skip_folds:
        continue

    if reset and train:
        ml_utils.prepare_train_dir(config.get_result_dir(i_fold, training_dir), backup=backup)
        config.make_dirs()

    for class_id in train_dataset.get_classes():

        model_dir = config.get_result_dir(i_fold, training_dir, config.Artifacts.MODEL)
        output_dir = config.get_result_dir(i_fold, training_dir, config.Artifacts.OUTPUT)
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
            
            trainer = ml_diff.DiffusionModel(batch_size=batch_size,
                                             n_epochs = n_epochs,
                                             lr = lr)
            errors = trainer.fit(data = class_train_dataset)#, export_progress_file=training_sample_mp4)

            trainer.save(trainer_file)

            batchs = range(1, errors.shape[0] + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(batchs, errors[:,0], label='Discriminator Real Error')
            # plt.plot(batchs, errors[:,1], label='Discriminator Fake Error')
            # plt.plot(batchs, errors[:,2], label='Generator Error')
            plt.xlabel('Batchs')
            plt.ylabel('Error')
            plt.title('Generator and Discriminator Errors per Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(training_loss_file)
            plt.close()

        if evaluate:

            if not os.path.exists(trainer_file):
                continue

            trainer = ml_model.Serializable.load(trainer_file)
            images = trainer.generate_images(n_samples=n_samples)

            for index, image in enumerate(images):
                image_file = os.path.join(output_dir, f'{index}.png')
                image.save(image_file)

        if one_class_only:
            break

    if one_fold_only:
        break