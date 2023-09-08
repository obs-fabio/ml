import os
import numpy as np
import tqdm
import typing
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.gan.trainer as ml_trainer
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils
import app.config as config

training_dict = {
    # train properties
    'id': config.Training.GAN,
    'n_epochs': 4096,
    'batch_size': 64,
    'n_samples': 20,
    # generator properties
    'latent_space_dim': 128,
    'g_lr': 1e-3,
    'g_n_cycles': 1,
    'g_internal_dims': [256, 512],
    # discriminator properties
    'd_lr': 1e-4,
    'd_n_cycles': 1,
    'd_internal_dims': [512, 256],
    'd_dropout': 0.2,
}

def run(reset: bool = True,
        backup: bool = False,
        train: bool = True,
        evaluate: bool = True,
        one_fold_only: bool = False,
        one_class_only: bool = False,
        skip_folds: typing.List[int] = [],
        skip_class: typing.List[str] = [],
        ):

    if reset and train:
        ml_utils.prepare_train_dir(config.get_result_dir(training_dict['id']), backup=backup)
        config.make_dirs()

    for i_fold, (train_dataset, _, _) in tqdm.tqdm(enumerate(config.get_dataset_loro()), leave=False):

        if i_fold in skip_folds:
            continue

        for class_id in tqdm.tqdm(train_dataset.get_classes()):

            if class_id in skip_class:
                continue

            model_dir = config.get_result_dir(training_dict['id'], config.Artifacts.MODEL, i_fold)
            output_dir = config.get_result_dir(training_dict['id'], config.Artifacts.OUTPUT, i_fold)
            output_dir = os.path.join(output_dir, class_id)
            os.makedirs(output_dir, exist_ok=True)

            trainer_file = os.path.join(model_dir, f'{class_id}_model.plk')
            training_loss_file = os.path.join(model_dir, f'{class_id}_acc_history.png')
            training_sample_mp4 = os.path.join(model_dir, f'{class_id}_sample.mp4')

            if train:

                if os.path.exists(trainer_file):
                    continue

                class_train_dataset = train_dataset.filt_dataset(class_id)

                trainer = ml_trainer.GAN_trainer(n_epochs = training_dict['n_epochs'],
                                                    batch_size = training_dict['batch_size'],
                                                    latent_space_dim = training_dict['latent_space_dim'],
                                                    g_lr = training_dict['g_lr'],
                                                    g_n_cycles = training_dict['g_n_cycles'],
                                                    g_internal_dims = training_dict['g_internal_dims'],
                                                    d_lr = training_dict['d_lr'],
                                                    d_n_cycles = training_dict['d_n_cycles'],
                                                    d_internal_dims = training_dict['d_internal_dims'],
                                                    d_dropout = training_dict['d_dropout'],)

                errors = trainer.fit(data = class_train_dataset, export_progress_file=training_sample_mp4)
                trainer.save(trainer_file)

                epochs = range(128)

                plt.figure(figsize=(10, 6))
                for i in range(errors.shape[1]):
                    erro = errors[:,i].reshape(-1)
                    n = training_dict['n_epochs']//128
                    erro = erro.reshape(n, 128)
                    erro = np.mean(erro, axis=0)
                    plt.plot(epochs, erro, label=f'Disc Error({i})')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                plt.savefig(training_loss_file)
                plt.close()

            if evaluate:

                if not os.path.exists(trainer_file):
                    continue

                trainer = ml_model.Serializable_model.load(trainer_file)
                images = trainer.generate_images(n_samples=training_dict['n_samples'])

                for index, image in enumerate(images):
                    image_file = os.path.join(output_dir, f'{index}.png')
                    image.save(image_file)

            if one_class_only:
                break

        if one_fold_only:
            break


if __name__ == "__main__":
    ml_utils.print_available_device()
    config.make_dirs()
    run()