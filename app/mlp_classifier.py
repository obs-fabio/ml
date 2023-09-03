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
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import app.config as config


source_model = config.Training.GAN
# trainings = [config.Training.CLASSIFIER_MLP_REAL, config.Training.CLASSIFIER_MLP_SYNTHETIC, config.Training.CLASSIFIER_MLP_JOINT]
trainings = [config.Training.CLASSIFIER_MLP_REAL, config.Training.CLASSIFIER_MLP_SYNTHETIC]
source_synthetics = [config.Training.GAN, config.Training.GANBIN]

batch_size=32
n_epochs=64
lr=1e-3

reset=True
backup=True
train = True
evaluate = True
compare = True
one_fold_only = True

skip_folds = [0, 1, 2]

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

        if training == config.Training.CLASSIFIER_MLP_REAL:

            source_model_dir = config.get_result_dir(i_fold, source_model, config.Artifacts.MODEL)
            classifier_dir = config.get_result_dir(i_fold, training, config.Artifacts.MODEL)

            if train:
                for id, class_id in enumerate(class_list):

                    classifier_file = os.path.join(classifier_dir, f'{class_id}_model.plk')
                    classifier_loss = os.path.join(classifier_dir, f'{class_id}_loss_history.png')
                    if os.path.exists(classifier_file):
                        continue

                    trainer_file = os.path.join(source_model_dir, f'{class_id}_model.plk')
                    trainer = ml_model.Serializable.load(trainer_file)
                    classifier = trainer.d_model
                    classifier.reset_output_layer()

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

            if evaluate:

                classifier_output_dir = config.get_result_dir(i_fold, training, config.Artifacts.OUTPUT)
                classifier_result_train = os.path.join(classifier_output_dir, f'model({training})_eval({training})_train.csv')
                classifier_result_val = os.path.join(classifier_output_dir, f'model({training})_eval({training})_val.csv')
                classifier_result_test = os.path.join(classifier_output_dir, f'model({training})_eval({training})_test.csv')

                if os.path.exists(classifier_result_train) and \
                    os.path.exists(classifier_result_val) and \
                    os.path.exists(classifier_result_test):
                    continue

                classifiers = []
                for class_id in class_list:
                    classifier_file = os.path.join(classifier_dir, f'{class_id}_model.plk')
                    classifiers.append(ml_model.Serializable.load(classifier_file))

                df_train = ml_model.eval_specialist_classifiers(
                        classes = class_list,
                        classifiers = classifiers,
                        dataset=train_dataset,
                        linearize_data=True
                    )
                df_val = ml_model.eval_specialist_classifiers(
                        classes = class_list,
                        classifiers = classifiers,
                        dataset=val_dataset,
                        linearize_data=True
                    )
                df_test = ml_model.eval_specialist_classifiers(
                        classes = class_list,
                        classifiers = classifiers,
                        dataset=test_dataset,
                        linearize_data=True
                    )

                df_train.to_csv(classifier_result_train, index=False)
                df_val.to_csv(classifier_result_val, index=False)
                df_test.to_csv(classifier_result_test, index=False)

                for source_synthetic in source_synthetics:

                    classifier_result_train = os.path.join(classifier_output_dir, f'model({training})_eval({str(source_synthetic)})_train.csv')
                    classifier_result_val = os.path.join(classifier_output_dir, f'model({training})_eval({str(source_synthetic)})_val.csv')
                    classifier_result_test = os.path.join(classifier_output_dir, f'model({training})_eval({str(source_synthetic)})_test.csv')

                    if os.path.exists(classifier_result_train) and \
                        os.path.exists(classifier_result_val) and \
                        os.path.exists(classifier_result_test):
                        continue

                    syn_train_dataset, syn_val_dataset, syn_test_dataset = ml_data.load_synthetic_dataset(
                            dir = config.get_result_dir(i_fold, source_synthetic, config.Artifacts.OUTPUT),
                            transform = train_dataset.transform
                        )

                    df_train = ml_model.eval_specialist_classifiers(
                            classes = class_list,
                            classifiers = classifiers,
                            dataset=syn_train_dataset,
                            linearize_data=True
                        )
                    df_val = ml_model.eval_specialist_classifiers(
                            classes = class_list,
                            classifiers = classifiers,
                            dataset=syn_val_dataset,
                            linearize_data=True
                        )
                    df_test = ml_model.eval_specialist_classifiers(
                            classes = class_list,
                            classifiers = classifiers,
                            dataset=syn_test_dataset,
                            linearize_data=True
                        )

                    df_train.to_csv(classifier_result_train, index=False)
                    df_val.to_csv(classifier_result_val, index=False)
                    df_test.to_csv(classifier_result_test, index=False)

                
        if training == config.Training.CLASSIFIER_MLP_SYNTHETIC:

            source_model_dir = config.get_result_dir(i_fold, source_model, config.Artifacts.MODEL)
            classifier_dir = config.get_result_dir(i_fold, training, config.Artifacts.MODEL)
            
            for source_synthetic in source_synthetics:

                if train:
                    for class_id in class_list:

                        classifier_file = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_model.plk')
                        classifier_loss = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_loss_history.png')

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
                        plt.yscale('log')
                        plt.title(f'Classifier {class_id} from real data')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(classifier_loss)
                        plt.close()

                        classifier.save(classifier_file)

                if evaluate:

                    classifier_output_dir = config.get_result_dir(i_fold, training, config.Artifacts.OUTPUT)
                    classifier_result_train = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval(real)_train.csv')
                    classifier_result_val = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval(real)_val.csv')
                    classifier_result_test = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval(real)_test.csv')

                    if not (os.path.exists(classifier_result_train) and \
                        os.path.exists(classifier_result_val) and \
                        os.path.exists(classifier_result_test)):


                        classifiers = []
                        for class_id in class_list:
                            classifier_file = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_model.plk')
                            classifiers.append(ml_model.Serializable.load(classifier_file))


                        df_train = ml_model.eval_specialist_classifiers(
                                classes = class_list,
                                classifiers = classifiers,
                                dataset=train_dataset,
                                linearize_data=True
                            )
                        df_val = ml_model.eval_specialist_classifiers(
                                classes = class_list,
                                classifiers = classifiers,
                                dataset=val_dataset,
                                linearize_data=True
                            )
                        df_test = ml_model.eval_specialist_classifiers(
                                classes = class_list,
                                classifiers = classifiers,
                                dataset=test_dataset,
                                linearize_data=True
                            )

                        df_train.to_csv(classifier_result_train, index=False)
                        df_val.to_csv(classifier_result_val, index=False)
                        df_test.to_csv(classifier_result_test, index=False)

                    for source_synthetic2 in source_synthetics:

                        classifier_result_train = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval({str(source_synthetic2)})_train.csv')
                        classifier_result_val = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval({str(source_synthetic2)})_val.csv')
                        classifier_result_test = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval({str(source_synthetic2)})_test.csv')

                        if os.path.exists(classifier_result_train) and \
                            os.path.exists(classifier_result_val) and \
                            os.path.exists(classifier_result_test):
                            continue

                        classifiers = []
                        for class_id in class_list:
                            classifier_file = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_model.plk')
                            classifiers.append(ml_model.Serializable.load(classifier_file))

                        syn_train_dataset, syn_val_dataset, syn_test_dataset = ml_data.load_synthetic_dataset(
                                dir = config.get_result_dir(i_fold, source_synthetic2, config.Artifacts.OUTPUT),
                                transform = train_dataset.transform
                            )

                        df_train = ml_model.eval_specialist_classifiers(
                                classes = class_list,
                                classifiers = classifiers,
                                dataset=syn_train_dataset,
                                linearize_data=True
                            )
                        df_val = ml_model.eval_specialist_classifiers(
                                classes = class_list,
                                classifiers = classifiers,
                                dataset=syn_val_dataset,
                                linearize_data=True
                            )
                        df_test = ml_model.eval_specialist_classifiers(
                                classes = class_list,
                                classifiers = classifiers,
                                dataset=syn_test_dataset,
                                linearize_data=True
                            )

                        df_train.to_csv(classifier_result_train, index=False)
                        df_val.to_csv(classifier_result_val, index=False)
                        df_test.to_csv(classifier_result_test, index=False)


    if one_fold_only:
        break

if compare:

    for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), desc="Folds", leave=False):

        if i_fold in skip_folds:
            continue

        subsets = [
            # 'train',
            'val',
            # 'test'
        ]

        print("####")

        output_dir = config.get_result_dir(i_fold, config.Training.PLOTS)

        for subset in subsets:

            for source in trainings:

                classifier_output_dir = config.get_result_dir(i_fold, source, config.Artifacts.OUTPUT)

                for model, eval in itertools.product([source] + source_synthetics, repeat=2):

                    classifier_result_train = os.path.join(classifier_output_dir, f'model({str(model)})_eval({str(eval)})_{subset}.csv')
                    if not os.path.exists(classifier_result_train):
                        continue

                    df = pd.read_csv(classifier_result_train, index_col=None)

                    real_columns = df.columns[:4]
                    predict_columns = df.columns[4:8]

                    real_to_index = {label: class_list[index] for index, label in enumerate(real_columns)}
                    predict_to_index = {label: class_list[index] for index, label in enumerate(predict_columns)}

                    df['real'] = df[real_columns].idxmax(axis=1)
                    df['predict'] = df[predict_columns].idxmax(axis=1)
                    df['real'] = df['real'].map(real_to_index)
                    df['predict'] = df['predict'].map(predict_to_index)

                    cm = sklearn.confusion_matrix(df['real'], df['predict'], labels=class_list)
                    disp = sklearn.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_list)
                    disp.plot(cmap='Blues', values_format='d', xticks_rotation='vertical')

                    cm_file = os.path.join(output_dir, f'cm_fold({i_fold})_model({str(model)})_eval({str(eval)})_{subset}.png')
                    plt.savefig(cm_file)
                    plt.close()

        # classifier_result_test = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval({str(source_synthetic2)})_test.csv')


        if one_fold_only:
            break
