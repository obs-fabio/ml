import os, tqdm
import itertools
import numpy as np
import pandas as pd
import sklearn.metrics as sklearn
import torch
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt
from tabulate import tabulate

import labsonar_ml.synthesizers.gan.trainer as ml_gan
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import app.config as config


training = config.Training.MLP_CLASSIFIER
source_synthetics = [config.Training.GAN, config.Training.GANSPE]

output_path = "./result/tables"

batch_size=32
n_epochs=128
lr=1e-3

reset=True
backup=True
train = True
evaluate = True
compare = True

skip_folds = []

ml_utils.print_available_device()
config.make_dirs()
config.set_seed()

class_list = ['A','B','C','D']


os.makedirs(output_path, exist_ok=True)


subsets = [
    # 'train',
    'val',
    # 'test'
]

def plot_confusion_matrix(cms, class_list, title, filename):

    print("cms: ", len(cms))

    mean_cm = np.mean(cms, axis=0)
    std_cm = np.std(cms, axis=0)

    print("mean_cm: ", mean_cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(mean_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=45)
    plt.yticks(tick_marks, class_list)

    for i in range(mean_cm.shape[0]):
        for j in range(mean_cm.shape[1]):
            plt.text(j, i, f"{mean_cm[i, j]:.2f}\n±{std_cm[i, j]:.2f}", horizontalalignment="center", color="white" if mean_cm[i, j] > mean_cm.max() / 2 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()


def get_metrics(model, eval, subset):

    cms = []
    f1s = []
    recalls = []
    accs = []  

    for i_fold in range(10):

        classifier_output_dir = config.get_result_dir(i_fold, training, config.Artifacts.OUTPUT)
        classifier_result_train = os.path.join(classifier_output_dir, f'model({str(model)})_eval({str(eval)})_{subset}.csv')
        print(classifier_result_train)
        if not os.path.exists(classifier_result_train):
            continue
        print("\t ok")

        df = pd.read_csv(classifier_result_train, index_col=None)

        cm, f1, recall, acc = ml_model.eval_metrics(df, class_list)

        cms.append(cm)
        f1s.append(f1)
        recalls.append(recall)
        accs.append(acc)

    plot_confusion_matrix(cms, class_list, "", os.path.join(output_path, f'model({str(model)})_eval({str(eval)})_{subset}.png'))


    return cms, f1s, recalls, accs


for subset in subsets:


    #tabela de comparação do modelo real analisando as sinteses dos modelos

    # table = [[''] * (4) for _ in range(len(source_synthetics)+1)]

    # for s, source in enumerate(source_synthetics):

    #     cms, f1s, recalls, accs = get_metrics(model=config.Training.CLASSIFIER_MLP_REAL,
    #                                             eval=source,
    #                                             subset=subset)

    #     mean_cm = np.mean(cms, axis=0)
    #     std_cm = np.std(cms, axis=0)
    #     mean_f1 = np.mean(f1s)
    #     std_f1 = np.std(f1s)
    #     mean_recall = np.mean(recalls)
    #     std_recall = np.std(recalls)
    #     mean_acc = np.mean(accs)
    #     std_acc = np.std(accs)

    #     table[s + 1][1] = f'${mean_f1} \pm {std_f1}$'
    #     table[s + 1][2] = f'${mean_recall} \pm {std_recall}$'
    #     table[s + 1][3] = f'${mean_acc} \pm {std_acc}$'


    # with open(os.path.join(output_path, f'real_model_{subset}.tex'), 'w') as f:
    #     f.write(tabulate(table, headers='firstrow', tablefmt='latex_raw'))


    # table = [[''] * (4) for _ in range(len(source_synthetics)+1)]

    # table[0][1] = f'f1-score'
    # table[0][2] = f'recall'
    # table[0][3] = f'acc'

    #tabela de comparação do modelo sintético analisando nos dados reais
    table = [[''] * (4) for _ in range(len(source_synthetics)+1)]
    for s, source in enumerate(source_synthetics):

        cms, f1s, recalls, accs = get_metrics(model=source,
                                                eval=config.Training.CLASSIFIER_MLP_REAL,
                                                subset=subset)

        mean_cm = np.mean(cms, axis=0)
        std_cm = np.std(cms, axis=0)
        mean_f1 = np.mean(f1s) * 100
        std_f1 = np.std(f1s) * 100
        mean_recall = np.mean(recalls) * 100
        std_recall = np.std(recalls) * 100
        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100

        table[s + 1][0] = str(source)
        table[s + 1][1] = f'${mean_f1:.2f} \pm {std_f1:.2f}$'
        table[s + 1][2] = f'${mean_recall:.2f} \pm {std_recall:.2f}$'
        table[s + 1][3] = f'${mean_acc:.2f} \pm {std_acc:.2f}$'

    with open(os.path.join(output_path, f'synthetic_model_{subset}.tex'), 'w') as f:
        f.write(tabulate(table, headers='firstrow', tablefmt='latex_raw'))


    # table = np.array( [[''] * (4) for _ in range(len(source_synthetics)+1)] )

    # for s, source in enumerate(source_synthetics):

    #     cms, f1s, recalls, accs = get_metrics(model=source,
    #                                             eval=config.Training.CLASSIFIER_MLP_REAL,
    #                                             subset=subset)

    #     mean_cm = np.mean(cms, axis=0)
    #     std_cm = np.std(cms, axis=0)
    #     mean_f1 = np.mean(f1s)
    #     std_f1 = np.std(f1s)
    #     mean_recall = np.mean(recalls)
    #     std_recall = np.std(recalls)
    #     mean_acc = np.mean(accs)
    #     std_acc = np.std(accs)

    #     table[s + 1,1]  = f'${mean_f1} \pm {std_f1}$'
    #     table[s + 1,2]  = f'${mean_recall} \pm {std_recall}$'
    #     table[s + 1,3]  = f'${mean_acc} \pm {std_acc}$'


    # with open(os.path.join(output_path, f'synthetic_model_{subset}.tex'), 'w') as f:
    #     f.write(tabulate(table, headers='firstrow', tablefmt='latex_raw'))

