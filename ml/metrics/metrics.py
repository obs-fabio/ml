from enum import Enum
import numpy as np
import os
import math
import itertools
import json
from tabulate import tabulate
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import tikzplotlib as tikz
import seaborn as sea

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
    
    def as_label(self, pt_br=True):

        pt_br_labels = {
            __class__.AUC: "AUC",
            __class__.F1: "F1 score",
            __class__.RECALL: "Recall",
            __class__.SENSITIVITY: "Sensitividade",
            __class__.PRECISION: "Precisão",
            __class__.ACCURACY: "Acurária",
            __class__.SPECIFICITY: "Especificidade",
            __class__.FALSE_ALARM: "Falso Alarme",
            __class__.SP_INDEX: "Índice SP",
            __class__.LOG_LOSS: "Perda logarítmica",
            __class__.MSE: "Erro médio quadrático"
        }
        en_labels = {
            __class__.AUC: "AUC",
            __class__.F1: "F1",
            __class__.RECALL: "Recall",
            __class__.SENSITIVITY: "Sensitivity",
            __class__.PRECISION: "Precision",
            __class__.ACCURACY: "Accuracy",
            __class__.SPECIFICITY: "Specificity",
            __class__.FALSE_ALARM: "False Alarm",
            __class__.SP_INDEX: "SP index",
            __class__.LOG_LOSS: "Log Loss",
            __class__.MSE: "MSE"
        }

        if pt_br:
            return pt_br_labels[self]
        return en_labels[self]

    @staticmethod
    def auc(y_target,y_predict):
        return skmetrics.roc_auc_score(y_target, y_predict) * 100

    @staticmethod
    def f1(y_target, y_predict):
        return skmetrics.f1_score(y_target, y_predict) * 100

    @staticmethod
    def recall(y_target, y_predict):
        return skmetrics.recall_score(y_target, y_predict) * 100

    @staticmethod
    def sensitivity(y_target, y_predict):
        return skmetrics.recall_score(y_target, y_predict) * 100

    @staticmethod
    def precision(y_target, y_predict):
        return skmetrics.precision_score(y_target, y_predict) * 100

    @staticmethod
    def accuracy(y_target, y_predict):
        return skmetrics.accuracy_score(y_target, y_predict) * 100

    @staticmethod
    def specificity(y_target, y_predict):
        tn, fp, fn, tp = Confusion_matrix.relative(y_target, y_predict)
        return tn / (tn + fp) * 100

    @staticmethod
    def false_alarm(y_target, y_predict):
        tn, fp, fn, tp = Confusion_matrix.relative(y_target, y_predict)
        return fp / (tn + fp) * 100

    @staticmethod
    def sp_index(y_target, y_predict):
        spec = __class__.specificity(y_target, y_predict)
        sens = __class__.sensitivity(y_target, y_predict)
        return np.sqrt(spec*sens)*np.mean([spec, sens]) * 100

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
        if self in [__class__.F1,__class__.AUC, __class__.ACCURACY, __class__.PRECISION, __class__.RECALL, __class__.SENSITIVITY, __class__.SPECIFICITY, __class__.SP_INDEX]:
            return lambda x: np.max(x)
        elif self in [__class__.FALSE_ALARM, __class__.LOG_LOSS, __class__.MSE]:
            return lambda x: np.min(x)
        else:
            raise NotImplementedError("best result for " + str(self))

    def get_best_index_func(self):
        if self in [__class__.F1,__class__.AUC, __class__.ACCURACY, __class__.PRECISION, __class__.RECALL, __class__.SENSITIVITY, __class__.SPECIFICITY, __class__.SP_INDEX]:
            return lambda x: np.argmax(x)
        elif self in [__class__.FALSE_ALARM, __class__.LOG_LOSS, __class__.MSE]:
            return lambda x: np.argmin(x)
        else:
            raise NotImplementedError("best result for " + str(self))
        
    def get_best_eval(self):
        if self in [__class__.F1,__class__.AUC, __class__.ACCURACY, __class__.PRECISION, __class__.RECALL, __class__.SENSITIVITY, __class__.SPECIFICITY, __class__.SP_INDEX]:
            return lambda x,y: x>y
        elif self in [__class__.FALSE_ALARM, __class__.LOG_LOSS, __class__.MSE]:
            return lambda x,y: x<y
        else:
            raise NotImplementedError("best result for " + str(self))
        
    def get_best_improve(self):
        if self in [__class__.F1,__class__.AUC, __class__.ACCURACY, __class__.PRECISION, __class__.RECALL, __class__.SENSITIVITY, __class__.SPECIFICITY, __class__.SP_INDEX]:
            return lambda x,y: x+y
        elif self in [__class__.FALSE_ALARM, __class__.LOG_LOSS, __class__.MSE]:
            return lambda x,y: x-y
        else:
            raise NotImplementedError("best result for " + str(self))


class Cross_validation_compiler():

    def __init__(self):
        self._score_dict = {'id':[],'model_path':[]}

        for metric in Metric:
            self._score_dict[str(metric)] = []
        for cm in Confusion_matrix:
            self._score_dict[str(cm)] = []

    def add(self, metrics, model_path, id):
        self._score_dict['model_path'].append(model_path)
        self._score_dict['id'].append(id)

        for metric, score in metrics.items():
            self._score_dict[str(metric)].append(score)

    def eval(self, y_target, y_predict, model_path, id, decision_threshold=0.5):
        self.add(Metric.eval_scores(y_target, y_predict, decision_threshold), model_path, id)

    def get_scores(self, metric):
        return self._score_dict[str(metric)]
    
    def get_mean(self, metric):
        return np.mean(self._score_dict[str(metric)])

    def get_std(self, metric):
        return np.std(self._score_dict[str(metric)])

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

    def get_best_model(self, metric):        
        return Base.load(self.get_by_id(self.get_best_id(metric))["model_path"])

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

    @staticmethod
    def str_format(values, n_samples=None, tex_format=False):
        decimal_places = int(math.log10(math.sqrt(n_samples))+1)
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

    def metric_as_str(self, metric, n_samples=None, tex_format=False):
        return __class__.str_format(self._score_dict[str(metric)], n_samples, tex_format)

    def as_str(self, metrics, n_samples=None, tex_format=False):
        ret = ['' for _ in metrics]
        for i, metric in enumerate(metrics):            
            decimal_places = int(math.log10(math.sqrt(n_samples))+1)
            ret[i] = self.metric_as_str(metric, n_samples, tex_format)
        return ret

    def cm_as_table(self, n_samples=None, tex_format=False, cm_state=Confusion_matrix.RELATIVE, pt_br=True):
        tns, fps, fns, tps = self.get_cm(cm_state)

        table = [[''] * (4) for _ in range(4)]
        table[0][2] = 'Predito' if pt_br else 'Predict'
        table[1][2] = 'Não' if pt_br else 'No'
        table[1][3] = 'Sim' if pt_br else 'Yes'
        table[2][0] = 'Real'
        table[2][1] = 'Não' if pt_br else 'No'
        table[3][1] = 'Sim' if pt_br else 'Yes'

        table[2][2] = __class__.str_format(tns, n_samples, tex_format)
        table[2][3] = __class__.str_format(fps, n_samples, tex_format)
        table[3][2] = __class__.str_format(fns, n_samples, tex_format)
        table[3][3] = __class__.str_format(tps, n_samples, tex_format)

        if pt_br:
            table[2][2] = table[2][2].replace(".",",")
            table[2][3] = table[2][3].replace(".",",")
            table[3][2] = table[3][2].replace(".",",")
            table[3][3] = table[3][3].replace(".",",")

        return table

    def cm_as_str(self, n_samples=None, cm_state=Confusion_matrix.RELATIVE, pt_br=True):
        return __class__.table_to_str(
            self.cm_as_table(n_samples, False, cm_state, pt_br)
        )

    def cm_save_tex(self, filename, n_samples=None, cm_state=Confusion_matrix.RELATIVE, pt_br=True):
        __class__.table_to_tex(
            self.cm_as_table(n_samples, True, cm_state, pt_br),
            filename
        )

    def __str__(self):
        return json.dumps(self._score_dict, indent=4)


class Grid_compiler():
    default_metrics = [Metric.F1, Metric.FALSE_ALARM, Metric.PRECISION, Metric.AUC]
    default_n_samples = 60
    default_pt_br = True
    default_fig_size = (10, 7)

    def __init__(self, n_samples = None, pt_br = None):
        self.cv_dict = {}
        self.param_dict = {}
        self.params = None
        self.n_samples = Grid_compiler.default_n_samples if n_samples is None else n_samples
        self.pt_br = Grid_compiler.default_pt_br if pt_br is None else pt_br

    def add_fold(self, params, cv):
        if not isinstance(params, dict):
            params = {'': params}

        params_hash = hash(tuple(params.items()))
        self.params = params.keys()

        if not params_hash in self.cv_dict:
            self.cv_dict[params_hash]  = {
                'params': params,
                'fold': cv,
            }
            self.param_dict[params_hash] = params
        else:
            self.cv_dict[params_hash]['fold'] = cv

    def add(self, params, metrics, model_path, id):
        params_hash = hash(tuple(params.items()))
        self.params = params.keys()

        if not params_hash in self.cv_dict:
            self.cv_dict[params_hash]  = {
                'params': params,
                'fold': Cross_validation_compiler(),
            }
            self.param_dict[params_hash] = params

        self.cv_dict[params_hash]['fold'].add(metrics, model_path, id)

    def eval(self, y_target, y_predict, params, model_path, id, decision_threshold=0.5):
        self.add(params, Metric.eval_scores(y_target, y_predict, decision_threshold), model_path, id)

    def get_fold(self, params):
        params_hash = hash(tuple(params.items()))
        return self.cv_dict[params_hash]['fold']

    @staticmethod
    def pack_to_hashs(param_pack):
        if param_pack is None:
            return []
        valid_hash_ids = []
        combinations = list(itertools.product(*param_pack.values()))
        for combination in combinations:
            parameter_dict = dict(zip(param_pack.keys(), combination))
            param_hash = hash(tuple(parameter_dict.items()))
            valid_hash_ids.append(param_hash)
        return valid_hash_ids

    @staticmethod
    def param_to_string(params):
        if len(params.keys()) == 1:
            for key, value in params.items():
                return str(value)
        id = ""
        for key, value in params.items():
            if id != "":
                id = id + "_"
            id = id + key + "[" + str(value) + "]"
        return id

    def get_bests(self, param_pack=None):
        valid_hash_ids = Grid_compiler.pack_to_hashs(param_pack)

        best = {}
        for metric in Metric:
            best[metric] = {}

        i = 0
        for hash_id, cv_dict in self.cv_dict.items():

            if param_pack is not None and hash_id not in valid_hash_ids:
                continue

            for metric in Metric:
                update_best = i == 0

                eval_best = metric.get_best_eval()

                if not update_best and eval_best(
                        cv_dict['fold'].get_mean(metric),
                        best[metric]["mean"]):
                    update_best = True

                if update_best:
                    best[metric]["mean"] = cv_dict['fold'].get_mean(metric)
                    best[metric]["std"] = cv_dict['fold'].get_std(metric)
                    best[metric]["index"] = i
                    best[metric]["params"] = cv_dict["params"]
                    best[metric]["id"] = cv_dict['fold'].get_best_id(metric)
            i = i +1

        return best

    def get_best_param(self, metric, param_pack=None):
        return self.get_bests(param_pack)[metric]['params']

    def get_best_fold(self, metric, param_pack=None):
        return self.get_fold(self.get_best_param(metric, param_pack))

    def get_best_model(self, metric, param_pack=None):
        return self.get_fold(self.get_best_param(metric, param_pack)).get_best_model(metric)

    def as_table(self, param_pack=None, metrics=None, tex_format=False):
        if metrics is None:
            metrics = Grid_compiler.default_metrics

        if self.params is None:
            raise UnboundLocalError("Grid_compiler evaluated without adding metrics")

        if param_pack is None:
            table = [[''] * (len(self.params) + len(metrics)) for _ in range(len(self.cv_dict)+1)]
        else:
            valid_hash_ids = Grid_compiler.pack_to_hashs(param_pack)
            table = [[''] * (len(self.params) + len(metrics)) for _ in range(len(valid_hash_ids)+1)]

        j = 0
        for p, param in enumerate(self.params):
            table[0][j] = str(param).replace('_', ' ') if self.pt_br else str(param).replace('_', ' ')
            j = j + 1

        for metric in metrics:
            table[0][j] = metric.as_label(self.pt_br)
            j = j + 1

        bests = self.get_bests(param_pack)

        i = 1
        for hash_id, cv_dict in self.cv_dict.items():
            j = 0

            if param_pack is not None and hash_id not in valid_hash_ids:
                continue

            for param_id, param_value in cv_dict['params'].items():
                table[i][j] = str(param_value)
                j = j+1

            for metric in metrics:

                if not tex_format:
                    table[i][j] = cv_dict['fold'].metric_as_str(metric, self.n_samples)
                else:
                    value_str = cv_dict['fold'].metric_as_str(metric, self.n_samples, True)

                    if i == bests[metric]['index']  + 1:
                        table[i][j] = "$\\mathbf{" + value_str[1:-1] + "}$"
                    else:
                        eval_best = metric.get_best_eval()
                        eval_improve = metric.get_best_improve()

                        test_score = cv_dict['fold'].get_by_id(cv_dict['fold'].get_best_id(metric))
                        if eval_best(test_score[str(metric)],
                                     eval_improve(bests[metric]["mean"], -bests[metric]["std"])):
                            table[i][j] = value_str
                        else:
                            table[i][j] = "\\textcolor{red}{" + value_str + "}"
                j = j+1

            i = i+1

        if self.pt_br:
            for row in table:
                for i in range(len(row)):
                    row[i] = row[i].replace(".", ",")
        return table

    def as_str(self, param_pack=None, metrics=None):
        return Cross_validation_compiler.table_to_str(
            self.as_table(param_pack, metrics, False)
        )

    def save_tex(self, filename, param_pack=None, metrics=None):
        Cross_validation_compiler.table_to_tex(
            self.as_table(param_pack, metrics, True),
            filename
        )

    def get_scores(self, metric, param_pack=None):
        data = []
        names = []
        valid_hash_ids = Grid_compiler.pack_to_hashs(param_pack)
        for hash_id, cv_dict in self.cv_dict.items():
            if param_pack is not None and hash_id not in valid_hash_ids:
                continue

            data.append(cv_dict['fold'].get_scores(metric))
            names.append(Grid_compiler.param_to_string(cv_dict['params']))
        return data, names

    def boxplot(self, metric, filepath=None, param_pack=None):
        data, names = self.get_scores(metric, param_pack)

        fig, ax = plt.subplots(figsize = Grid_compiler.default_fig_size)
        ax = fig.add_axes([0, 0, 1, 1])
        bp = ax.boxplot(data, vert=False)
        plt.yticks(ticks=range(1, len(names) + 1), labels=names)

        if filepath is not None:
            filename, extension = os.path.splitext(filepath)
            print(extension)
            if extension == ".tex":
                tikz.save(filepath)
            else:
                plt.savefig(filepath)
        else:
            plt.show()
        plt.close()

    def violinplot(self, metric, filepath=None, param_pack=None):
        data, names = self.get_scores(metric, param_pack)

        sea.set(style="white")
        fig, ax = plt.subplots(figsize = Grid_compiler.default_fig_size)
        sea.violinplot(data=data, orient='h', ax=ax, color='skyblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)

        if filepath is not None:
            filename, extension = os.path.splitext(filepath)
            print(extension)
            if extension == ".tex":
                tikz.save(filepath)
            else:
                plt.savefig(filepath)
        else:
            plt.show()
        plt.close()

    def __str__ (self):
        return self.as_str()

if __name__ == "__main__":

    print("--- Test Metrics ---")
    y_target = np.random.randint(2, size=100)
    # y_predict = np.random.randint(2, size=100)
    y_predict = np.random.rand(100)

    metric_dict = Metric.eval_scores(y_target, y_predict)

    for metric, score in metric_dict.items():
        print('\t',metric, ':', score)
        
    for metric in Metric:
        print('\t',metric.as_label())

    # print("\n--- Metrics_compiler ---")

    # compiler = Cross_validation_compiler()
    # for i in range(5):
    #     y_target = np.random.randint(2, size=100)
    #     y_predict = np.random.rand(100)

    #     compiler.eval(y_target, y_predict, id={'fold': i,
    #                                            'model': 'filename'})

    # print("as str: ", compiler.as_str([Metric.F1, Metric.FALSE_ALARM]))
    # print("as str: ", compiler.as_str([Metric.F1, Metric.FALSE_ALARM], 100))
    # print("as tex: ", compiler.as_str([Metric.F1, Metric.FALSE_ALARM], tex_format=True))

    # print("cm_as_str: ", compiler.cm_as_str())
    # print("cm_abs_as_str: ", compiler.cm_as_str(cm_state=Confusion_matrix.ABSOLUTE))
    # # compiler.cm_as_tex("a.tex")
    # # compiler.cm_as_tex("b.tex",cm_state=Confusion_matrix.ABSOLUTE)

    # best_id = compiler.get_best_id(Metric.F1)
    # print("best: ", best_id)
    # print("best scores: ", compiler.get_by_id(best_id))
    # # print()

    print("\n--- Metrics_compiler ---")

    # Grid_compiler.default_n_samples=100
    # Grid_compiler.default_pt_br=False
    grid = Grid_compiler()

    params = {
        'drop out': ["None","0.2","0.4"],
        'regularização': [0.8,0.2,0.1],
    }
    combinations = list(itertools.product(*params.values()))

    for combination in combinations:
        parameter_dict = dict(zip(params.keys(), combination))

        for ifold in range(5):
            y_target = np.random.randint(2, size=100)
            y_predict = np.random.rand(100) + parameter_dict['regularização']

            grid.eval(y_target,
                    y_predict,
                    params=parameter_dict,
                    model_path='filename',
                    id= ifold)
            
    print(grid.as_str())
    # grid.save_tex("a.tex")
    print("best param F1: ", grid.get_best_param(Metric.F1))
    print("best fold F1: ", grid.get_best_fold(Metric.F1))


    params = {
        'drop out': ["0.2","0.4"],
        'regularização': [0.2,0.1],
    }
    # grid.save_tex("b.tex",params)
    print(grid.as_str(params))

    print("--- subgrid ---")
    grid2 = Grid_compiler()
    metrics = [Metric.F1, Metric.LOG_LOSS, Metric.AUC]
    for metric in metrics:
        grid2.add_fold(str(metric),
                grid.get_best_fold(metric))
    
    print(grid2.as_str(metrics=metrics))

    print("\n--- boxplot ---")
    grid2.boxplot(Metric.AUC, "boxplot.png")
    grid2.boxplot(Metric.AUC, "boxplot.tex")
    grid2.violinplot(Metric.AUC, "violinplot.png")
    grid2.violinplot(Metric.AUC, "violinplot.tex")



