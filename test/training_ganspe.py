import os
import tqdm
import typing
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.gan.trainer as ml_trainer
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import app.config as config

trainings_dict = [
    {
        # train properties
        'id': config.Training.GAN,
        'n_epochs': 128,
        'batch_size': 64,
        'n_samples': 20,
        # generator properties
        'latent_space_dim': 128,
        'g_lr': 5e-4,
        'g_n_cycles': 1,
        'g_internal_dims': [64, 256],
        # discriminator properties
        'd_lr': 3e-4,
        'd_n_cycles': 1,
        'd_internal_dims': [256, 64],
        'd_dropout': 0.2,
        # specialist discriminator properties
        'sd_lr': 5e-4,
        'sd_internal_dims': [256, 64],
        'sd_dropout': 0.2,
        'sd_bins': [i for i in range(5, 16)],
        'sd_reg_factor': 1,
    },
]

def run(reset: bool = True,
        backup: bool = False,
        train: bool = True,
        evaluate: bool = True,
        one_fold_only: bool = True,
        one_class_only: bool = True,
        skip_folds: typing.List[int] = [],
        skip_class: typing.List[str] = [],
        ):

    data_path = "./data"
    output_path = "./test_result/ganspe"

    bins = []
    for i in range(5,16):
        bins.append(i)

    if reset and train:
        ml_utils.prepare_train_dir(output_path, backup=backup)
        config.make_dirs()

    for training_dict in tqdm.tqdm(trainings_dict):

        training_dict['sd_bins'] = bins

        for class_id in tqdm.tqdm(range(10)):

            if class_id in skip_class:
                continue

            model_dir = os.path.join(output_path, str(config.Artifacts.MODEL))
            output_dir = os.path.join(output_path, str(config.Artifacts.OUTPUT))
            output_dir = os.path.join(output_dir, str(class_id))
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            trainer_file = os.path.join(model_dir, f'{class_id}_model.plk')
            training_loss_file = os.path.join(model_dir, f'{class_id}_acc_history.png')
            training_evolution_file = os.path.join(model_dir, f'{class_id}_sample.gif')

            if train:

                if os.path.exists(trainer_file):
                    continue

                class_train_dataset = ml_data.get_mnist_dataset_as_specialist(data_path, class_id)

                trainer = ml_trainer.SPECGAN_trainer(n_epochs = training_dict['n_epochs'],
                                                    batch_size = training_dict['batch_size'],
                                                    latent_space_dim = training_dict['latent_space_dim'],
                                                    g_lr = training_dict['g_lr'],
                                                    g_n_cycles = training_dict['g_n_cycles'],
                                                    g_internal_dims = training_dict['g_internal_dims'],
                                                    d_lr = training_dict['d_lr'],
                                                    d_n_cycles = training_dict['d_n_cycles'],
                                                    d_internal_dims = training_dict['d_internal_dims'],
                                                    d_dropout = training_dict['d_dropout'],
                                                    sd_lr = training_dict['sd_lr'],
                                                    sd_internal_dims = training_dict['sd_internal_dims'],
                                                    sd_dropout = training_dict['sd_dropout'],
                                                    sd_bins = training_dict['sd_bins'],
                                                    sd_reg_factor = training_dict['sd_reg_factor'])

                errors = trainer.fit(data = class_train_dataset, export_progress_file=training_evolution_file)
                trainer.save(trainer_file)

                epochs = range(errors.shape[0])

                plt.figure(figsize=(10, 6))
                for i in range(errors.shape[1]):
                    plt.plot(epochs, errors[:,i], label=f'Disc Error({i})')
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



if __name__ == "__main__":
    ml_utils.print_available_device()
    config.make_dirs()
    run()