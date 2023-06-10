from enum import Enum
import numpy as np
import os
import math
import itertools
import json
from tabulate import tabulate
import sklearn.metrics as skmetrics

from ml.models.base import Base


class Confusion_matrix(Enum):
    RELATIVE = 1
    ABSOLUTE = 2

    @staticmethod
    def relative(y_target, y_predict):
        tn, fp, fn, tp = skmetrics.confusion_matrix(y_target, y_predict, normalize='true').ravel()
        return (tn, fp, fn, tp)

    @staticmethod
    def absolute(y_target, y_predict):
        tn, fp, fn, tp = __class__.relative(y_target, y_predict)
        ps = np.sum(y_target)
        ns = np.sum(1-y_target)
        return (tn * ns, fp * ns, fn * ps, tp * ps)

    def eval(self, y_target, y_predict):
        return getattr(self.__class__, self.name.lower())(y_target, y_predict)


class Metric(Enum): 
    AUC = 1
    F1 = 2
    RECALL = 3
    SENSITIVITY = 3
    PRECISION = 4
    ACCURACY = 5
    SPECIFICITY = 6
    FALSE_ALARM = 7
    SP_INDEX = 8
    LOG_LOSS = 9
    MSE = 10

    def __str__(self):
        return str(self.name).split('.')[-1].lower()

    @staticmethod
    def auc(y_target,y_predict):
        return skmetrics.roc_auc_score(y_target, y_predict)

    @staticmethod
    def f1(y_target, y_predict):
        return skmetrics.f1_score(y_target, y_predict)

    @staticmethod
    def recall(y_target, y_predict):
        return skmetrics.recall_score(y_target, y_predict)

    @staticmethod
    def sensitivity(y_target, y_predict):
        return skmetrics.recall_score(y_target, y_predict)

    @staticmethod
    def precision(y_target, y_predict):
        return skmetrics.precision_score(y_target, y_predict)

    @staticmethod
    def accuracy(y_target, y_predict):
        return skmetrics.accuracy_score(y_target, y_predict)

    @staticmethod
    def specificity(y_target, y_predict):
        tn, fp, fn, tp = Confusion_matrix.relative(y_target, y_predict)
        return tn / (tn + fp)

    @staticmethod
    def false_alarm(y_target, y_predict):
        tn, fp, fn, tp = Confusion_matrix.relative(y_target, y_predict)
        return fp / (tn + fp)

    @staticmethod
    def sp_index(y_target, y_predict):
        spec = __class__.specificity(y_target, y_predict)
        sens = __class__.sensitivity(y_target, y_predict)
        return np.sqrt(spec*sens)*np.mean([spec, sens])

    @staticmethod
    def log_loss(y_target, y_predict):
        return skmetrics.log_loss(y_target, y_predict)

    @staticmethod
    def mse(y_target, y_predict):
        return skmetrics.mean_squared_error(y_target, y_predict)

    def eval(self, y_target, y_predict):
        return getattr(self.__class__, self.name.lower())(y_target, y_predict)

    @staticmethod
    def eval_scores(y_target, y_predict, decision_threshold=0.5):
        if len(y_target.shape) == 2 and y_target.shape[1] == 1:
            y_target = np.squeeze(y_target)
        if len(y_predict.shape) == 2 and y_predict.shape[1] == 1:
            y_predict = np.squeeze(y_predict)

        metrics = {}
        for metric in __class__:
            try:
                metrics[str(metric)] = metric.eval(y_target, y_predict)
            except:
                metrics[str(metric)] = metric.eval(y_target, (y_predict > decision_threshold).astype(int))

        for cm in Confusion_matrix:
            metrics[str(cm)] = cm.eval(y_target, (y_predict > decision_threshold).astype(int))

        return metrics

    def get_best_func(self):
        if self in [__class__.AUC, __class__.ACCURACY, __class__.PRECISION, __class__.RECALL, __class__.SENSITIVITY, __class__.SPECIFICITY]:
            return lambda x: np.max(x)
        elif self in [__class__.F1, __class__.FALSE_ALARM, __class__.SP_INDEX, __class__.LOG_LOSS, __class__.MSE]:
            return lambda x: np.min(x)
        else:
            raise NotImplementedError("best result for " + str(self))

    def get_best_index_func(self):
        if self in [__class__.AUC, __class__.ACCURACY, __class__.PRECISION, __class__.RECALL, __class__.SENSITIVITY, __class__.SPECIFICITY]:
            return lambda x: np.argmax(x)
        elif self in [__class__.F1, __class__.FALSE_ALARM, __class__.SP_INDEX, __class__.LOG_LOSS, __class__.MSE]:
            return lambda x: np.argmin(x)
        else:
            raise NotImplementedError("best result for " + str(self))


class Cross_validation_compiler():

    def __init__(self):
        self._score_dict = {'id':[]}

        for metric in Metric:
            self._score_dict[str(metric)] = []
        for cm in Confusion_matrix:
            self._score_dict[str(cm)] = []

    def add(self, metrics, id=None):
        self._score_dict['id'].append(id)

        for metric, score in metrics.items():
            self._score_dict[str(metric)].append(score)

    def eval(self, y_target, y_predict, id=None, decision_threshold=0.5):
        self.add(Metric.eval_scores(y_target, y_predict, decision_threshold), id)

    def get_by_id(self, id):
        index = self._score_dict['id'].index(id)

        output_dict = {}
        for key, score_list in self._score_dict.items():
            if key == 'id':
                continue
            output_dict[key] = score_list[index]
        return output_dict

    def get_best_id(self, metric):
        best_index = metric.get_best_index_func()
        index = best_index(self._score_dict[str(metric)])
        return self._score_dict['id'][index]

    def get_cm(self, cm_state=Confusion_matrix.RELATIVE):
        tns = []
        fps = []
        fns = []
        tps = []
        for cm in self._score_dict[str(cm_state)]:
            tns.append(cm[0] * (100 if cm_state == Confusion_matrix.RELATIVE else 1))
            fps.append(cm[1] * (100 if cm_state == Confusion_matrix.RELATIVE else 1))
            fns.append(cm[2] * (100 if cm_state == Confusion_matrix.RELATIVE else 1))
            tps.append(cm[3] * (100 if cm_state == Confusion_matrix.RELATIVE else 1))
        return (tns, fps, fns, tps)

    def __str__(self):
        return json.dumps(self._score_dict, indent=4)

    @staticmethod
    def str_format(values, num_samples=60, tex_format=False):
        decimal_places = int(math.log10(math.sqrt(num_samples))+1)
        if tex_format:
            return '${:.{}f} \\pm {:.{}f}$'.format(
                np.mean(values),
                decimal_places,
                np.std(values),
                decimal_places
            )
        return '{:.{}f} \u00B1 {:.{}f}'.format(
            np.mean(values),
            decimal_places,
            np.std(values),
            decimal_places
        )

    @staticmethod
    def table_to_str(table) -> str:
        num_columns = len(table[0])
        column_widths = [max(len(str(row[i])) for row in table) for i in range(num_columns)]

        formatted_rows = []
        for row in table:
            formatted_row = [str(row[i]).ljust(column_widths[i]) for i in range(num_columns)]
            formatted_rows.append('  '.join(formatted_row))

        return '\n'.join(formatted_rows)
    
    @staticmethod
    def table_to_tex(table, filename):
        with open(filename, 'w') as f:
            f.write(tabulate(table, headers='firstrow', tablefmt='latex_raw'))

    def metric_as_str(self, metric, num_samples=60, tex_format=False):
        return __class__.str_format(self._score_dict[str(metric)], num_samples, tex_format)

    def as_str(self, metrics, num_samples=60, tex_format=False):
        ret = ['' for _ in metrics]
        for i, metric in enumerate(metrics):            
            decimal_places = int(math.log10(math.sqrt(num_samples))+1)
            ret[i] = self.metric_as_str(metric, num_samples, tex_format)
        return ret

    def cm_as_table(self, num_samples=60, tex_format=False, cm_state=Confusion_matrix.RELATIVE, pt_br=True):
        tns, fps, fns, tps = self.get_cm(cm_state)

        table = [[''] * (4) for _ in range(4)]
        table[0][2] = 'Predito' if pt_br else 'Predict'
        table[1][2] = 'Não' if pt_br else 'No'
        table[1][3] = 'Sim' if pt_br else 'Yes'
        table[2][0] = 'Real'
        table[2][1] = 'Não' if pt_br else 'No'
        table[3][1] = 'Sim' if pt_br else 'Yes'

        table[2][2] = __class__.str_format(tns, num_samples, tex_format)
        table[2][3] = __class__.str_format(fps, num_samples, tex_format)
        table[3][2] = __class__.str_format(fns, num_samples, tex_format)
        table[3][3] = __class__.str_format(tps, num_samples, tex_format)

        return table

    def cm_as_str(self, num_samples=60, cm_state=Confusion_matrix.RELATIVE, pt_br=True):
        return __class__.table_to_str(
            self.cm_as_table(num_samples, False, cm_state, pt_br)
        )

    def cm_as_tex(self, filename, num_samples=60, cm_state=Confusion_matrix.RELATIVE, pt_br=True):
        __class__.table_to_tex(
            self.cm_as_table(num_samples, True, cm_state, pt_br),
            filename
        )


if __name__ == "__main__":

    print("--- Test Metrics ---")
    y_target = np.random.randint(2, size=100)
    # y_predict = np.random.randint(2, size=100)
    y_predict = np.random.rand(100)

    metric_dict = Metric.eval_scores(y_target, y_predict)

    for metric, score in metric_dict.items():
        print('\t',metric, ':', score)

    print("--- Metrics_compiler ---")

    compiler = CV_compiler()
    for i in range(5):
        y_target = np.random.randint(2, size=100)
        y_predict = np.random.rand(100)

        compiler.eval(y_target, y_predict, id={'fold': i,
                                               'model': 'filename'})

    print("as str: ", compiler.as_str([Metric.F1, Metric.FALSE_ALARM]))
    print("as str: ", compiler.as_str([Metric.F1, Metric.FALSE_ALARM], 100))
    print("as tex: ", compiler.as_str([Metric.F1, Metric.FALSE_ALARM], tex_format=True))

    print("cm_as_str: ", compiler.cm_as_str())
    print("cm_abs_as_str: ", compiler.cm_as_str(cm_state=Confusion_matrix.ABSOLUTE))
    # compiler.cm_as_tex("a.tex")
    # compiler.cm_as_tex("b.tex",cm_state=Confusion_matrix.ABSOLUTE)

    best_id = compiler.get_best_id(Metric.F1)
    print("best: ", best_id)
    print("best scores: ", compiler.get_by_id(best_id))
    # print()

