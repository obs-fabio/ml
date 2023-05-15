from enum import Enum
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import itertools
from tabulate import tabulate

def confusion_matrix(y_target, y_predict):
    tn, fp, fn, tp = cm(y_target, y_predict, normalize='true').ravel()
    return (tn, fp, fn, tp)

def sensitivity(y_target, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_target, y_predict)
    sensitivity = tp / (tp+fn)
    return sensitivity

def specificity(y_target, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_target, y_predict)
    specificity = tn / (tn+fp)
    return specificity

def auc(y_target,y_predict):
    return roc_auc_score(y_target, y_predict)

def sp_index(y_target, y_predict):
    spec = sensitivity(y_target, y_predict)
    sens = sensitivity(y_target, y_predict)
    return np.sqrt(spec*sens)*np.mean([spec, sens])

def acc(y_target, y_predict):
    return accuracy_score(y_target, y_predict)

class Scores(Enum):
    AUC = 1
    SENSITIVITY = 2
    SPECIFICITY = 3
    SP_INDEX = 4
    ACC = 5
    CONFUSION_MATRIX = 6

def get_scores(y_target, y_predict):
    return {
        Scores.AUC.value: auc(y_target, y_predict),
        Scores.SENSITIVITY.value: sensitivity(y_target, y_predict),
        Scores.SPECIFICITY.value: specificity(y_target, y_predict),
        Scores.SP_INDEX.value: sp_index(y_target, y_predict),
        Scores.ACC.value: acc(y_target, y_predict),
        Scores.CONFUSION_MATRIX.value: (confusion_matrix(y_target, y_predict)),
    }

class Manager():

    def __init__(self):
        self.score_dict = {}
        self.param_dict = {}
        self.n_params = 0
        self.values = {}

    def add_score(self, params, score):
        params_hash = hash(tuple(params.items()))

        if not params_hash in self.score_dict:
            self.score_dict[params_hash] = []
            self.param_dict[params_hash] = params
            self.n_params = params.keys()

            if not params_hash in self.values:
                self.values[params_hash] = {
                    'params': params,
                    'scores': {}
                }

                for s in Scores:
                    self.values[params_hash]['scores'][s] = []

            for s in Scores:
                self.values[params_hash]['scores'][s].append(score[str(s.value)]*100)

        self.score_dict[params_hash].append(score)

    def export_score_tex(self, filename, valid_scores = None):
        if valid_scores == None:
            valid_scores = [score for score in Scores if score.value <= Scores.ACC.value]


        table = [[None] * (len(self.n_params) + len(valid_scores)) for _ in range(len(self.score_dict)+1)]

        j = 0
        for p, param in enumerate(self.n_params):
            table[0][j] = str(param).replace('_', ' ')
            j = j + 1

        for score in valid_scores:
            score_str = str(score).split(".")[-1].lower().replace('_', ' ')
            table[0][j] = score_str
            j = j + 1

        i = 1
        for hash, value in self.values.items():
            j = 0

            for param_id, param_value in value['params'].items():
                table[i][j] = str(param_value)
                j = j+1

            for score, values in value['scores'].items():
                if not score in valid_scores:
                    continue

                table[i][j] = '${:.2f} \\pm {:.2f}$'.format(np.mean(values), np.std(values))
                j = j+1

            i = i+1

        with open(filename, 'w') as f:
            f.write(tabulate(table, headers='firstrow', floatfmt=".2f", tablefmt='latex_raw'))