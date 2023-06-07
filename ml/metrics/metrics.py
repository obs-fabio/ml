from enum import Enum
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import itertools
from tabulate import tabulate
from ml.models.base import Base

def confusion_matrix(y_target, y_predict):
    tn, fp, fn, tp = cm(y_target, y_predict, normalize='true').ravel()
    return (tn, fp, fn, tp)

def abs_confusion_matrix(y_target, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_target, y_predict)
    ps = np.sum(y_target)
    ns = np.sum(1-y_target)
    return (tn * ns, fp * ns, fn * ps, tp * ps)

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
    ABS_CONFUSION_MATRIX = 7

def get_scores(y_target, y_predict):
    if len(y_target.shape) == 2 and y_target.shape[1] == 1:
        y_target = np.squeeze(y_target)
    if len(y_predict.shape) == 2 and y_predict.shape[1] == 1:
        y_predict = np.squeeze(y_predict)

    return {
        str(Scores.AUC): auc(y_target, y_predict),
        str(Scores.SENSITIVITY): sensitivity(y_target, y_predict),
        str(Scores.SPECIFICITY): specificity(y_target, y_predict),
        str(Scores.SP_INDEX): sp_index(y_target, y_predict),
        str(Scores.ACC): acc(y_target, y_predict),
        str(Scores.CONFUSION_MATRIX): (confusion_matrix(y_target, y_predict)),
        str(Scores.ABS_CONFUSION_MATRIX): (abs_confusion_matrix(y_target, y_predict)),
    }

class Manager():

    def __init__(self):
        self.dict = {}
        self.param_dict = {}
        self.params = None

    def add_score(self, params, score, model_path):
        params_hash = hash(tuple(params.items()))
        self.params = params.keys()

        if not params_hash in self.dict:
            self.dict[params_hash]  = {
                'params': params,
                'scores': {},
                'models': []
            }
            self.param_dict[params_hash] = params

            for s in Scores:
                self.dict[params_hash]['scores'][s] = []

        for s in Scores:
            if s == Scores.CONFUSION_MATRIX:
                value = [v*100 for v in score[str(s)]]
                self.dict[params_hash]['scores'][s].append(value)
            else:
                self.dict[params_hash]['scores'][s].append(score[str(s)]*100)
            self.dict[params_hash]['models'].append(model_path)

    def __str__(self) -> str:
        table = self.get_table()

        num_columns = len(table[0])
        column_widths = [max(len(str(row[i])) for row in table) for i in range(num_columns)]

        formatted_rows = []
        for row in table:
            formatted_row = [str(row[i]).ljust(column_widths[i]) for i in range(num_columns)]
            formatted_rows.append('  '.join(formatted_row))

        return '\n'.join(formatted_rows)
    
    def get_max_scores(self):
        max = {}
        for score in Scores:
            if score.value <= Scores.ACC.value:
                max[str(score)] = {
                    "mean": -1,
                    "std": -1,
                    "index": 0,
                    "params": None,
                    "model": None
                }

        i = 0
        for hash, dict_item in self.dict.items():
            for score, values in dict_item['scores'].items():
                if score.value <= Scores.ACC.value:
                    if max[str(score)]["mean"] < np.mean(values):
                        max[str(score)]["mean"] = np.mean(values)
                        max[str(score)]["std"] = np.std(values)
                        max[str(score)]["index"] = i
                        max[str(score)]["params"] = dict_item["params"]
                        max[str(score)]["model"] = dict_item["models"][i]
            i = i +1

        return max
    
    def get_best_model(self, score_criterion: str = Scores.ACC.value):
        folds_scores = self.get_max_scores()

        return Base.load(folds_scores[score_criterion]["model"]).model

    def get_table(self, valid_scores = None, latex_format = False):

        if valid_scores == None:
            valid_scores = [score for score in Scores if score.value <= Scores.ACC.value]

        table = [[None] * (len(self.params) + len(valid_scores)) for _ in range(len(self.dict)+1)]

        j = 0
        for p, param in enumerate(self.params):
            table[0][j] = str(param).replace('_', ' ')
            j = j + 1

        for score in valid_scores:
            score_str = str(score).split(".")[-1].lower().replace('_', ' ')
            table[0][j] = score_str
            j = j + 1

        max = self.get_max_scores()

        i = 1
        for hash, dict_item in self.dict.items():
            j = 0

            for param_id, param_value in dict_item['params'].items():
                table[i][j] = str(param_value)
                j = j+1

            for score, values in dict_item['scores'].items():
                if not score in valid_scores:
                    continue
                
                mean = np.mean(values)
                std = np.std(values)

                if latex_format:
                    if i == max[str(score)]["index"] + 1:
                        table[i][j] = '\\textbf{' + '${:.2f} \\pm {:.2f}$'.format(mean, std) + "}"
                    elif (mean + std) >= (max[str(score)]["mean"] - max[str(score)]["std"]):
                        table[i][j] = '\\textit{' + '${:.2f} \\pm {:.2f}$'.format(mean, std) + "}"
                    else:
                        table[i][j] = '${:.2f} \\pm {:.2f}$'.format(mean, std)
                else:
                    table[i][j] = '{:.2f} +- {:.2f}'.format(mean, std)
                j = j+1

            i = i+1

        return table

    def export_score_tex(self, filename, valid_scores = None):

        table = self.get_table(valid_scores=valid_scores, latex_format=True)

        with open(filename, 'w') as f:
            f.write(tabulate(table, headers='firstrow', floatfmt=".2f", tablefmt='latex_raw'))

    def get_table_confusion_matrix(self, params, latex_format=False):
        params_hash = hash(tuple(params.items()))
        tns = []
        fps = []
        fns = []
        tps = []
        for cm in self.dict[params_hash]['scores'][Scores.CONFUSION_MATRIX]:
            tns.append(cm[0])
            fps.append(cm[1])
            fns.append(cm[2])
            tps.append(cm[3])

        table = [[''] * (4) for _ in range(4)]
        table[0][2] = 'Predito'
        table[0][3] = ''
        table[1][2] = 'Não'
        table[1][3] = 'Sim'
        table[2][0] = 'Real'
        table[3][0] = ''
        table[2][1] = 'Não'
        table[3][1] = 'Sim'

        if latex_format:
            table[2][2] = '${:.2f} \\pm {:.2f}$'.format(np.mean(tns), np.std(tns))
            table[2][3] = '${:.2f} \\pm {:.2f}$'.format(np.mean(fps), np.std(fps))
            table[3][2] = '${:.2f} \\pm {:.2f}$'.format(np.mean(fns), np.std(fns))
            table[3][3] = '${:.2f} \\pm {:.2f}$'.format(np.mean(tps), np.std(tps))
        else:
            table[2][2] = '{:.2f} +- {:.2f}'.format(np.mean(tns), np.std(tns))
            table[2][3] = '{:.2f} +- {:.2f}'.format(np.mean(fps), np.std(fps))
            table[3][2] = '{:.2f} +- {:.2f}'.format(np.mean(fns), np.std(fns))
            table[3][3] = '{:.2f} +- {:.2f}'.format(np.mean(tps), np.std(tps))

        return table

    def str_confusion_matrix(self, params):
        table = self.get_table_confusion_matrix(params)

        num_columns = len(table[0])
        column_widths = [max(len(str(row[i])) for row in table) for i in range(num_columns)]

        formatted_rows = []
        for row in table:
            formatted_row = [str(row[i]).ljust(column_widths[i]) for i in range(num_columns)]
            formatted_rows.append('  '.join(formatted_row))

        return '\n'.join(formatted_rows)

    def export_confusion_matrix_tex(self, params, filename):
        table = self.get_table_confusion_matrix(params, latex_format=True)

        with open(filename, 'w') as f:
            f.write(tabulate(table, headers='firstrow', floatfmt=".2f", tablefmt='latex_raw'))

    def get_table_abs_confusion_matrix(self, params, latex_format=False):
        params_hash = hash(tuple(params.items()))
        tns = []
        fps = []
        fns = []
        tps = []
        for cm in self.dict[params_hash]['scores'][Scores.ABS_CONFUSION_MATRIX]:
            tns.append(cm[0])
            fps.append(cm[1])
            fns.append(cm[2])
            tps.append(cm[3])

        table = [[''] * (4) for _ in range(4)]
        table[0][2] = 'Predito'
        table[0][3] = ''
        table[1][2] = 'Não'
        table[1][3] = 'Sim'
        table[2][0] = 'Real'
        table[3][0] = ''
        table[2][1] = 'Não'
        table[3][1] = 'Sim'

        if latex_format:
            table[2][2] = '${:f} \\pm {:f}$'.format(np.mean(tns), np.std(tns))
            table[2][3] = '${:f} \\pm {:f}$'.format(np.mean(fps), np.std(fps))
            table[3][2] = '${:f} \\pm {:f}$'.format(np.mean(fns), np.std(fns))
            table[3][3] = '${:f} \\pm {:f}$'.format(np.mean(tps), np.std(tps))
        else:
            table[2][2] = '{:f} +- {:f}'.format(np.mean(tns), np.std(tns))
            table[2][3] = '{:f} +- {:f}'.format(np.mean(fps), np.std(fps))
            table[3][2] = '{:f} +- {:f}'.format(np.mean(fns), np.std(fns))
            table[3][3] = '{:f} +- {:f}'.format(np.mean(tps), np.std(tps))

        return table

    def str_abs_confusion_matrix(self, params):
        table = self.get_table_abs_confusion_matrix(params)

        num_columns = len(table[0])
        column_widths = [max(len(str(row[i])) for row in table) for i in range(num_columns)]

        formatted_rows = []
        for row in table:
            formatted_row = [str(row[i]).ljust(column_widths[i]) for i in range(num_columns)]
            formatted_rows.append('  '.join(formatted_row))

        return '\n'.join(formatted_rows)

    def export_abs_confusion_matrix_tex(self, params, filename):
        table = self.get_table_abs_confusion_matrix(params, latex_format=True)

        with open(filename, 'w') as f:
            f.write(tabulate(table, headers='firstrow', floatfmt=".2f", tablefmt='latex_raw'))