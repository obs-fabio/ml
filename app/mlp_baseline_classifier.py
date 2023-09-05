import os, tqdm
import itertools
import numpy as np
import pandas as pd
import sklearn.metrics as sklearn
import torch
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.model.mlp as ml_mlp
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import app.config as config

training = config.Training.MLP_CLASSIFIER
# source_synthetics = [config.Training.GAN, config.Training.GANSPE]
source_synthetics = [config.Training.GANSPE]

batch_size=32
n_epochs=64
lr=1e-3
n_neurons=8

reset=True
backup=True
train = True
evaluate = True
compare = True
one_fold_only = True
one_class_only = False

skip_folds = []
selected_class = ['D']

ml_utils.print_available_device()
config.make_dirs()
config.set_seed()

class_list = ['A','B','C','D']

if reset and train:
    ml_utils.prepare_train_dir(config.get_result_dir(0, training), backup=backup)
    config.make_dirs()

for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), desc="Folds", leave=False):

    if i_fold in skip_folds:
        continue

    image_shape = list((train_dataset.__getitem__(0)[0]).shape)

    classifier_dir = config.get_result_dir(i_fold, training, config.Artifacts.MODEL)
    classifier_output_dir = config.get_result_dir(i_fold, training, config.Artifacts.OUTPUT)

    if train:

        training_list = []

        # # treina o classificador com dados reais
        # for id, class_id in enumerate(class_list):

        #     classifier_file = os.path.join(classifier_dir, f'{class_id}_real_model.plk')
        #     classifier_loss = os.path.join(classifier_dir, f'{class_id}_real_loss_history.png')
        #     if os.path.exists(classifier_file):
        #         continue

        #     training_list.append(
        #         {
        #             'dataset': train_dataset,
        #             'class_id': class_id,
        #             'id': id,
        #             'classifier_file': classifier_file,
        #             'classifier_loss': classifier_loss,
        #         }
        #     )

        # treina o classificador com sintéticos
        for source_synthetic in source_synthetics:
            for id, class_id in enumerate(class_list):

                if not class_id in selected_class and one_class_only:
                    continue

                classifier_file = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_model.plk')
                classifier_loss = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_loss_history.png')

                if os.path.exists(classifier_file) and not one_class_only:
                    continue

                syn_train_dataset, syn_val_dataset, syn_test_dataset = ml_data.load_synthetic_dataset(
                        config.get_result_dir(i_fold, source_synthetic, config.Artifacts.OUTPUT),
                        train_dataset.transform)
                        
                training_list.append(
                    {
                        'dataset': syn_train_dataset,
                        'class_id': class_id,
                        'id': id,
                        'classifier_file': classifier_file,
                        'classifier_loss': classifier_loss,
                    }
                    )

        # executa os treinamentos
        for training_dict in training_list:

            classifier = ml_mlp.MLP(input_shape=image_shape, n_neurons=n_neurons)

            errors = ml_model.fit_specialist_classifier(
                    model=classifier,
                    dataset=training_dict['dataset'],
                    class_id=training_dict['id'],
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    lr=lr,
                    linearize_data=False
                )

            classifier.save(training_dict['classifier_file'])

            class_id = training_dict['class_id']

            batchs = range(1, errors.shape[0] + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(batchs, errors, label='Error')
            plt.xlabel('Batchs')
            plt.ylabel('Error')
            plt.yscale('log')
            plt.title(f'Class {class_id}')
            plt.legend()
            plt.grid(True)
            plt.savefig(training_dict['classifier_loss'])
            plt.close()

    if evaluate:

        evaluation_list = []

        # # classificador treinado nos dados reais avaliados nos dados sintéticos
        # for source_synthetic in source_synthetics:

        #     classifier_result_train = os.path.join(classifier_output_dir, f'model(real)_eval({str(source_synthetic)})_train.csv')
        #     classifier_result_val = os.path.join(classifier_output_dir, f'model(real)_eval({str(source_synthetic)})_val.csv')
        #     classifier_result_test = os.path.join(classifier_output_dir, f'model(real)_eval({str(source_synthetic)})_test.csv')

        #     if os.path.exists(classifier_result_train) and \
        #         os.path.exists(classifier_result_val) and \
        #         os.path.exists(classifier_result_test):
        #         continue

        #     syn_train_dataset, syn_val_dataset, syn_test_dataset = ml_data.load_synthetic_dataset(
        #             dir = config.get_result_dir(i_fold, source_synthetic, config.Artifacts.OUTPUT),
        #             transform = train_dataset.transform
        #         )

        #     evaluation_list.append(
        #         {
        #             'classifiers': [],
        #             'train_dataset': syn_train_dataset,
        #             'val_dataset': syn_val_dataset,
        #             'test_dataset': syn_test_dataset,
        #             'classifier_result_train' : classifier_result_train,
        #             'classifier_result_val' : classifier_result_val,
        #             'classifier_result_test' : classifier_result_test,
        #         }
        #     )

        #     for class_id in class_list:
        #         classifier_file = os.path.join(classifier_dir, f'{class_id}_real_model.plk')
        #         evaluation_list[-1]['classifiers'].append(ml_model.Serializable.load(classifier_file))

        # classificador treinado nos dados sintéticos avaliado nos dados reais
        for source_synthetic in source_synthetics:

            classifier_result_train = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval(real)_train.csv')
            classifier_result_val = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval(real)_val.csv')
            classifier_result_test = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval(real)_test.csv')

            if os.path.exists(classifier_result_train) and \
                os.path.exists(classifier_result_val) and \
                os.path.exists(classifier_result_test) and \
                not one_class_only:
                continue

            evaluation_list.append(
                {
                    'classifiers': [],
                    'train_dataset': train_dataset,
                    'val_dataset': val_dataset,
                    'test_dataset': test_dataset,
                    'classifier_result_train' : classifier_result_train,
                    'classifier_result_val' : classifier_result_val,
                    'classifier_result_test' : classifier_result_test,
                }
            )

            for class_id in class_list:
                classifier_file = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_model.plk')
                evaluation_list[-1]['classifiers'].append(ml_model.Serializable.load(classifier_file))

        # executa as avaliações
        for evaluation_dict in evaluation_list:

            df_train = ml_model.eval_specialist_classifiers(
                    classes = class_list,
                    classifiers = evaluation_dict['classifiers'],
                    dataset = evaluation_dict['train_dataset'],
                    linearize_data = False
                )
            df_val = ml_model.eval_specialist_classifiers(
                    classes = class_list,
                    classifiers = evaluation_dict['classifiers'],
                    dataset = evaluation_dict['val_dataset'],
                    linearize_data = False
                )
            df_test = ml_model.eval_specialist_classifiers(
                    classes = class_list,
                    classifiers = evaluation_dict['classifiers'],
                    dataset = evaluation_dict['test_dataset'],
                    linearize_data = False
                )

            df_train.to_csv(evaluation_dict['classifier_result_train'], index=False)
            df_val.to_csv(evaluation_dict['classifier_result_val'], index=False)
            df_test.to_csv(evaluation_dict['classifier_result_test'], index=False)

    if one_fold_only:
        break

if compare:

    output_dir = config.get_result_dir(0, config.Training.PLOTS)
    output_dir = os.path.join(output_dir, "classifiers")
    os.makedirs(output_dir, exist_ok=True)

    for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), desc="Folds", leave=False):

        if i_fold in skip_folds:
            continue

        subsets = [
            # 'train',
            'val',
            # 'test'
        ]

        classifier_output_dir = config.get_result_dir(i_fold, training, config.Artifacts.OUTPUT)

        for abs_filename in ml_utils.get_files(classifier_output_dir, ".csv"):

            path, rel_filename = os.path.split(abs_filename)
            filename, extension = os.path.splitext(rel_filename)
            cm_file = os.path.join(output_dir, "cm_" + filename + ".png")

            df = pd.read_csv(abs_filename, index_col=None)
            cm, f1, recall, acc = ml_model.eval_metrics(df, class_list)

            disp = sklearn.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_list)
            disp.plot(cmap='Blues', values_format='d', xticks_rotation='vertical')
            plt.savefig(cm_file)
            plt.close()


        if one_fold_only:
            break
