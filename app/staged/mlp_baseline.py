import os, tqdm
import itertools
import numpy as np
import pandas as pd
import sklearn.metrics as sklearn
import torch
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.gan.trainer as ml_gan
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.model.mlp as ml_mlp
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import app.config as config

n_neurons=8
batch_size=32
n_epochs=128
lr=1e-3

reset=False
backup=True
train = True
evaluate = True
compare = True
one_fold_only = False

skip_folds = []

ml_utils.print_available_device()
config.make_dirs()
config.set_seed()

class_list = ['A','B','C','D']

if reset and train:
    ml_utils.prepare_train_dir(config.get_result_dir(0, config.Training.BASELINE), backup=backup)
    config.make_dirs()

for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), desc="Folds", leave=False):

    if i_fold in skip_folds:
        continue

    image_shape = list((train_dataset.__getitem__(0)[0]).shape)

    baseline_model_dir = config.get_result_dir(i_fold, config.Training.BASELINE, config.Artifacts.MODEL)
    baseline_output_dir = config.get_result_dir(i_fold, config.Training.BASELINE, config.Artifacts.OUTPUT)

    classifiers = []

    for id, class_id in enumerate(class_list):

        model_file = os.path.join(baseline_model_dir, f'{class_id}_{str(config.Training.BASELINE)}_model.pkl')
        model_loss_file = os.path.join(baseline_model_dir, f'{class_id}_{str(config.Training.BASELINE)}_model_loss.png')

        if os.path.exists(model_file):
            classifiers.append(ml_model.Serializable.load(model_file))
            continue

        classifier = ml_mlp.MLP(input_shape=image_shape,
                                n_neurons=n_neurons)

        errors = ml_model.fit_specialist_classifier(
                model=classifier,
                dataset=train_dataset,
                class_id=id,
                batch_size=batch_size,
                n_epochs=n_epochs,
                lr=lr,
                linearize_data=True
            )

        classifier.save(model_file)
        classifiers.append(classifier)

        batchs = range(1, errors.shape[0] + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(batchs, errors, label='Error')
        plt.xlabel('Batchs')
        plt.ylabel('Error')
        plt.yscale('log')
        plt.title(f'Classifier {class_id} from real data')
        plt.legend()
        plt.grid(True)
        plt.savefig(model_loss_file)
        plt.close()

    output_file = os.path.join(baseline_output_dir, f'{str(config.Training.BASELINE)}_relevance.csv')

    if not os.path.exists(output_file):

        df_val = ml_model.eval_specialist_classifiers(
                classes = class_list,
                classifiers = classifiers,
                dataset=val_dataset,
                linearize_data=False
            )
        
        baseline_cm, baseline_f1, baseline_recall, baseline_acc = ml_model.eval_metrics(df_val, class_list)
        metrics = []
        #analise de impacto de feature na classificação
        for i in range(image_shape[2]): # y da imagem - 1 bin de frequência

            val_dataset.set_relevance_analisys(y_relevance=i)

            df_val = ml_model.eval_specialist_classifiers(
                    classes = class_list,
                    classifiers = classifiers,
                    dataset=val_dataset,
                    linearize_data=False
                )
            
            cm, f1, recall, acc = ml_model.eval_metrics(df_val, class_list)
            metrics.append([i, f1/baseline_f1, recall/baseline_recall, acc/baseline_acc])

        metrics = np.array(metrics)

        df = pd.DataFrame(metrics.T, ['Bin', 'F1', 'Recall', 'Accuracy']).T
        df.to_csv(output_file, index=False)

    else:
        df = pd.read_csv(output_file, index_col=None)

    files = []
    files.append(os.path.join(baseline_output_dir, f'{str(config.Training.BASELINE)}_f1.png'))
    files.append(os.path.join(baseline_output_dir, f'{str(config.Training.BASELINE)}_recall.png'))
    files.append(os.path.join(baseline_output_dir, f'{str(config.Training.BASELINE)}_acc.png'))

    index_files = []
    index_files.append(os.path.join(baseline_output_dir, f'{str(config.Training.BASELINE)}_f1.txt'))
    index_files.append(os.path.join(baseline_output_dir, f'{str(config.Training.BASELINE)}_recall.txt'))
    index_files.append(os.path.join(baseline_output_dir, f'{str(config.Training.BASELINE)}_acc.txt'))

    for i in range(1,4):
        plt.barh(df.iloc[:,0].to_numpy(), df.iloc[:,i].to_numpy(), color='skyblue')
        plt.xlabel('Valores')
        plt.ylabel('Categorias')
        plt.title('Gráfico de Barras Horizontais')
        plt.gca().invert_yaxis()
        plt.savefig(files[i-1])
        plt.close()

        aux = df.iloc[:,i].to_numpy()
        indexes = np.argsort(aux)[:30]

        np.savetxt(index_files[i-1], indexes, fmt='%d')

        print("####")
        print(df.columns[i])
        print(indexes)
        print(aux[indexes])


    if one_fold_only:
        break



#     if evaluate:

#         classifier_output_dir = config.get_result_dir(i_fold, training, config.Artifacts.OUTPUT)
#         classifier_result_train = os.path.join(classifier_output_dir, f'model({training})_eval({training})_train.csv')
#         classifier_result_val = os.path.join(classifier_output_dir, f'model({training})_eval({training})_val.csv')
#         classifier_result_test = os.path.join(classifier_output_dir, f'model({training})_eval({training})_test.csv')

#         if os.path.exists(classifier_result_train) and \
#             os.path.exists(classifier_result_val) and \
#             os.path.exists(classifier_result_test):
#             continue

#         classifiers = []
#         for class_id in class_list:
#             classifier_file = os.path.join(classifier_dir, f'{class_id}_model.plk')
#             classifiers.append(ml_model.Serializable.load(classifier_file))

#         df_train = ml_model.eval_specialist_classifiers(
#                 classes = class_list,
#                 classifiers = classifiers,
#                 dataset=train_dataset,
#                 linearize_data=True
#             )
#         df_val = ml_model.eval_specialist_classifiers(
#                 classes = class_list,
#                 classifiers = classifiers,
#                 dataset=val_dataset,
#                 linearize_data=True
#             )
#         df_test = ml_model.eval_specialist_classifiers(
#                 classes = class_list,
#                 classifiers = classifiers,
#                 dataset=test_dataset,
#                 linearize_data=True
#             )

#         df_train.to_csv(classifier_result_train, index=False)
#         df_val.to_csv(classifier_result_val, index=False)
#         df_test.to_csv(classifier_result_test, index=False)

#         for source_synthetic in source_synthetics:

#             classifier_result_train = os.path.join(classifier_output_dir, f'model({training})_eval({str(source_synthetic)})_train.csv')
#             classifier_result_val = os.path.join(classifier_output_dir, f'model({training})_eval({str(source_synthetic)})_val.csv')
#             classifier_result_test = os.path.join(classifier_output_dir, f'model({training})_eval({str(source_synthetic)})_test.csv')

#             if os.path.exists(classifier_result_train) and \
#                 os.path.exists(classifier_result_val) and \
#                 os.path.exists(classifier_result_test):
#                 continue

#             syn_train_dataset, syn_val_dataset, syn_test_dataset = ml_data.load_synthetic_dataset(
#                     dir = config.get_result_dir(i_fold, source_synthetic, config.Artifacts.OUTPUT),
#                     transform = train_dataset.transform
#                 )

#             df_train = ml_model.eval_specialist_classifiers(
#                     classes = class_list,
#                     classifiers = classifiers,
#                     dataset=syn_train_dataset,
#                     linearize_data=True
#                 )
#             df_val = ml_model.eval_specialist_classifiers(
#                     classes = class_list,
#                     classifiers = classifiers,
#                     dataset=syn_val_dataset,
#                     linearize_data=True
#                 )
#             df_test = ml_model.eval_specialist_classifiers(
#                     classes = class_list,
#                     classifiers = classifiers,
#                     dataset=syn_test_dataset,
#                     linearize_data=True
#                 )

#             df_train.to_csv(classifier_result_train, index=False)
#             df_val.to_csv(classifier_result_val, index=False)
#             df_test.to_csv(classifier_result_test, index=False)

        
# if training == config.Training.CLASSIFIER_MLP_SYNTHETIC:

#     source_model_dir = config.get_result_dir(i_fold, source_model, config.Artifacts.MODEL)
#     classifier_dir = config.get_result_dir(i_fold, training, config.Artifacts.MODEL)
    
#     for source_synthetic in source_synthetics:

#         if train:
#             for class_id in class_list:

#                 classifier_file = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_model.plk')
#                 classifier_loss = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_loss_history.png')

#                 if os.path.exists(classifier_file):
#                     continue

#                 syn_train_dataset, syn_val_dataset, syn_test_dataset = ml_data.load_synthetic_dataset(
#                         config.get_result_dir(i_fold, source_synthetic, config.Artifacts.OUTPUT),
#                         train_dataset.transform)

#                 trainer_file = os.path.join(source_model_dir, f'{class_id}_model.plk')
#                 trainer = ml_model.Serializable.load(trainer_file)
#                 classifier = trainer.d_model

#                 errors = ml_model.fit_specialist_classifier(
#                         model=classifier,
#                         dataset=syn_train_dataset,
#                         class_id=class_list.index(class_id),
#                         batch_size=batch_size,
#                         n_epochs=n_epochs,
#                         lr=lr,
#                         linearize_data=True
#                     )

#                 batchs = range(1, errors.shape[0] + 1)
#                 plt.figure(figsize=(10, 6))
#                 plt.plot(batchs, errors, label='Error')
#                 plt.xlabel('Batchs')
#                 plt.ylabel('Error')
#                 plt.yscale('log')
#                 plt.title(f'Classifier {class_id} from real data')
#                 plt.legend()
#                 plt.grid(True)
#                 plt.savefig(classifier_loss)
#                 plt.close()

#                 classifier.save(classifier_file)

#         if evaluate:

#             classifier_output_dir = config.get_result_dir(i_fold, training, config.Artifacts.OUTPUT)
#             classifier_result_train = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval(real)_train.csv')
#             classifier_result_val = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval(real)_val.csv')
#             classifier_result_test = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval(real)_test.csv')

#             if not (os.path.exists(classifier_result_train) and \
#                 os.path.exists(classifier_result_val) and \
#                 os.path.exists(classifier_result_test)):


#                 classifiers = []
#                 for class_id in class_list:
#                     classifier_file = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_model.plk')
#                     classifiers.append(ml_model.Serializable.load(classifier_file))


#                 df_train = ml_model.eval_specialist_classifiers(
#                         classes = class_list,
#                         classifiers = classifiers,
#                         dataset=train_dataset,
#                         linearize_data=True
#                     )
#                 df_val = ml_model.eval_specialist_classifiers(
#                         classes = class_list,
#                         classifiers = classifiers,
#                         dataset=val_dataset,
#                         linearize_data=True
#                     )
#                 df_test = ml_model.eval_specialist_classifiers(
#                         classes = class_list,
#                         classifiers = classifiers,
#                         dataset=test_dataset,
#                         linearize_data=True
#                     )

#                 df_train.to_csv(classifier_result_train, index=False)
#                 df_val.to_csv(classifier_result_val, index=False)
#                 df_test.to_csv(classifier_result_test, index=False)

#             for source_synthetic2 in source_synthetics:

#                 classifier_result_train = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval({str(source_synthetic2)})_train.csv')
#                 classifier_result_val = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval({str(source_synthetic2)})_val.csv')
#                 classifier_result_test = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval({str(source_synthetic2)})_test.csv')

#                 if os.path.exists(classifier_result_train) and \
#                     os.path.exists(classifier_result_val) and \
#                     os.path.exists(classifier_result_test):
#                     continue

#                 classifiers = []
#                 for class_id in class_list:
#                     classifier_file = os.path.join(classifier_dir, f'{class_id}_{str(source_synthetic)}_model.plk')
#                     classifiers.append(ml_model.Serializable.load(classifier_file))

#                 syn_train_dataset, syn_val_dataset, syn_test_dataset = ml_data.load_synthetic_dataset(
#                         dir = config.get_result_dir(i_fold, source_synthetic2, config.Artifacts.OUTPUT),
#                         transform = train_dataset.transform
#                     )

#                 df_train = ml_model.eval_specialist_classifiers(
#                         classes = class_list,
#                         classifiers = classifiers,
#                         dataset=syn_train_dataset,
#                         linearize_data=True
#                     )
#                 df_val = ml_model.eval_specialist_classifiers(
#                         classes = class_list,
#                         classifiers = classifiers,
#                         dataset=syn_val_dataset,
#                         linearize_data=True
#                     )
#                 df_test = ml_model.eval_specialist_classifiers(
#                         classes = class_list,
#                         classifiers = classifiers,
#                         dataset=syn_test_dataset,
#                         linearize_data=True
#                     )

#                 df_train.to_csv(classifier_result_train, index=False)
#                 df_val.to_csv(classifier_result_val, index=False)
#                 df_test.to_csv(classifier_result_test, index=False)



# if compare:

#     for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), desc="Folds", leave=False):

#         if i_fold in skip_folds:
#             continue

#         subsets = [
#             # 'train',
#             'val',
#             # 'test'
#         ]

#         print("####")

#         output_dir = config.get_result_dir(i_fold, config.Training.PLOTS)

#         for subset in subsets:

#             for source in trainings:

#                 classifier_output_dir = config.get_result_dir(i_fold, source, config.Artifacts.OUTPUT)

#                 for model, eval in itertools.product([source] + source_synthetics, repeat=2):

#                     classifier_result_train = os.path.join(classifier_output_dir, f'model({str(model)})_eval({str(eval)})_{subset}.csv')
#                     if not os.path.exists(classifier_result_train):
#                         continue

#                     df = pd.read_csv(classifier_result_train, index_col=None)

#                     real_columns = df.columns[:4]
#                     predict_columns = df.columns[4:8]

#                     real_to_index = {label: class_list[index] for index, label in enumerate(real_columns)}
#                     predict_to_index = {label: class_list[index] for index, label in enumerate(predict_columns)}

#                     df['real'] = df[real_columns].idxmax(axis=1)
#                     df['predict'] = df[predict_columns].idxmax(axis=1)
#                     df['real'] = df['real'].map(real_to_index)
#                     df['predict'] = df['predict'].map(predict_to_index)

#                     cm = sklearn.confusion_matrix(df['real'], df['predict'], labels=class_list)
# cm = confusion_matrix(df['real'], df['predict'], labels=class_list)

# # Calcula o F1-score, recall e acurácia
# f1_scores = classification_report(df['real'], df['predict'], labels=class_list, output_dict=True)
# accuracy = accuracy_score(df['real'], df['predict'])
#                     disp = sklearn.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_list)
#                     disp.plot(cmap='Blues', values_format='d', xticks_rotation='vertical')

#                     cm_file = os.path.join(output_dir, f'cm_fold({i_fold})_model({str(model)})_eval({str(eval)})_{subset}.png')
#                     plt.savefig(cm_file)
#                     plt.close()

#         # classifier_result_test = os.path.join(classifier_output_dir, f'model({str(source_synthetic)})_eval({str(source_synthetic2)})_test.csv')


#         if one_fold_only:
#             break
