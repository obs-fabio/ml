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
import app.config as config


source_model = config.Training.GAN
trainings = [config.Training.CLASSIFIER_MLP_REAL, config.Training.CLASSIFIER_MLP_SYNTHETIC, config.Training.CLASSIFIER_MLP_JOINT]
source_synthetics = [config.Training.GAN]

batch_size=32
n_epochs=64
lr=2e-4

reset=False
backup=True
train = True
evaluate = True
one_fold_only = True
one_class_only = False

skip_folds = []

ml_utils.print_available_device()
config.make_dirs()
config.set_seed()

class_list = ['A','B','C','D']

for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), desc="Folds", leave=False):

    if i_fold in skip_folds:
        continue

    for training in trainings:

        if reset and train:
            ml_utils.prepare_train_dir(config.get_result_dir(i_fold, training), backup=backup)
            config.make_dirs()

        if training == config.Training.CLASSIFIER_MLP_REAL:

            source_model_dir = config.get_result_dir(i_fold, source_model, config.Artifacts.MODEL)
            classifier_dir = config.get_result_dir(i_fold, training, config.Artifacts.MODEL)

            for id, class_id in enumerate(class_list):

                classifier_file = os.path.join(classifier_dir, f'{class_id}_model.plk')
                classifier_loss = os.path.join(classifier_dir, f'{class_id}_loss_history.png')
                if os.path.exists(classifier_file):
                    continue

                trainer_file = os.path.join(source_model_dir, f'{class_id}_model.plk')
                trainer = ml_model.Serializable.load(trainer_file)
                classifier = trainer.d_model

                errors = ml_model.fit_specialist_classifier(
                        model=classifier,
                        dataset=train_dataset,
                        class_id=id,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        lr=lr,
                        linearize_data=True
                    )

                classifier.save(classifier_file)

                batchs = range(1, errors.shape[0] + 1)
                plt.figure(figsize=(10, 6))
                plt.plot(batchs, errors, label='Error')
                plt.xlabel('Batchs')
                plt.ylabel('Error')
                plt.yscale('log')
                plt.title(f'Classifier {class_id} from real data')
                plt.legend()
                plt.grid(True)
                plt.savefig(classifier_loss)
                plt.close()

                if one_class_only:
                    break

        if training == config.Training.CLASSIFIER_MLP_SYNTHETIC:

            source_model_dir = config.get_result_dir(i_fold, source_model, config.Artifacts.MODEL)
            classifier_dir = config.get_result_dir(i_fold, training, config.Artifacts.MODEL)
            
            for source_synthetic in source_synthetics:

                for class_id in train_dataset.get_classes():

                    classifier_file = os.path.join(classifier_dir, f'{class_id}_{source_synthetic.name.lower()}_model.plk')
                    classifier_loss = os.path.join(classifier_dir, f'{class_id}_{source_synthetic.name.lower()}_loss_history.png')

                    if train:
                        if os.path.exists(classifier_file):
                            continue

                        syn_train_dataset, syn_val_dataset, syn_test_dataset = ml_data.load_synthetic_dataset(
                                config.get_result_dir(i_fold, source_synthetic, config.Artifacts.OUTPUT),
                                train_dataset.transform)


                        trainer_file = os.path.join(source_model_dir, f'{class_id}_model.plk')
                        trainer = ml_model.Serializable.load(trainer_file)
                        classifier = trainer.d_model

                        errors = ml_model.fit_specialist_classifier(
                                model=classifier,
                                dataset=syn_train_dataset,
                                class_id=class_list.index(class_id),
                                batch_size=batch_size,
                                n_epochs=n_epochs,
                                lr=lr,
                                linearize_data=True
                            )

                        batchs = range(1, errors.shape[0] + 1)
                        plt.figure(figsize=(10, 6))
                        plt.plot(batchs, errors, label='Error')
                        plt.xlabel('Batchs')
                        plt.ylabel('Error')
                        plt.title(f'Classifier {class_id} from real data')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(classifier_loss)
                        plt.close()

                        classifier.save(classifier_file)

                if evaluate:

                    classifier_output_dir = config.get_result_dir(i_fold, training, config.Artifacts.OUTPUT)
                    classifier_result_train = os.path.join(classifier_output_dir, f'{source_synthetic.name.lower()}_train.csv')
                    classifier_result_val = os.path.join(classifier_output_dir, f'{source_synthetic.name.lower()}_val.csv')
                    classifier_result_test = os.path.join(classifier_output_dir, f'{source_synthetic.name.lower()}_test.csv')

                    if os.path.exists(classifier_result_train) and \
                        os.path.exists(classifier_result_val) and \
                        os.path.exists(classifier_result_test):
                        continue

                    classifiers = []
                    for class_id in class_list:
                        classifier_file = os.path.join(classifier_dir, f'{class_id}_{source_synthetic.name.lower()}_model.plk')
                        classifiers.append(ml_model.Serializable.load(classifier_file))

                    df_train, df_val, df_test = ml_model.eval_specialist_classifiers(
                        classes = class_list,
                        classifiers = classifiers,
                        dir = config.get_result_dir(i_fold, source_synthetic, config.Artifacts.OUTPUT),
                        transform = train_dataset.transform,
                        linearize_data=True
                    )

                    df_train.to_csv(classifier_result_train, index=False)
                    df_val.to_csv(classifier_result_val, index=False)
                    df_test.to_csv(classifier_result_test, index=False)




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





