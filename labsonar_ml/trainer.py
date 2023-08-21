from enum import Enum
import os, json
import hashlib
import pickle
import pandas as pd
import numpy as np
import re
import joblib
import itertools
import time
import shutil
import datetime
from abc import ABC, abstractmethod

import sklearn.model_selection as sklmodel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, PowerTransformer

from sklearn.utils.class_weight import compute_class_weight

import labsonar_ml.models.base as ml
import labsonar_ml.metrics.metrics as metrics

def get_files(directory, extension):
	file_list = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.endswith(extension):
				file_list.append(os.path.join(root, file))
	return file_list

class Subdirs(Enum):
    CV = "cv"
    PIPELINE = "pipeline"
    MODEL = "model"
    PREDICTION = "predict"
    SCORE = "score"
    PLOT = "plot"

class Subsets(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3

class Experiment_results():

    def __init__(self, trn, val, test):
        self.trn = trn
        self.val = val
        self.test = test

    def __getitem__(self, subset):
        if subset == Subsets.TEST:
            return self.test
        if subset == Subsets.TRAINING:
            return self.trn
        if subset == Subsets.VALIDATION:
            return self.val
        raise UnboundLocalError('Implementation for get results ', subset, ' not found')

class Experiment():
    default_shuffle = True
    default_random_state = 42

    def __init__(self, config):
        self.id = config['id']
        self.input_file = config['input_file']
        self.input_columns = config['input_columns']
        self.output_column = config['output_column']
        self.test_subset = config['test_subset']
        self.n_folds = config['n_folds']
        self.n_repeats = config['n_repeats']
        self.pipeline = config['pipeline']
        self.model_constructor = config['model_constructor']
        self.model_grid = config['model_grid']
        self.base_output_path = config['base_output_path']
        self.split_subsets_done = False
        self.pipeline_fitting_done = False
        self.models_fitting_done = False
        self.models_prediction_done = False
        self.df_data = None
        self.df_target = None

        self.check_dirs(False)

    def check_data(self):
        if self.df_data is None:
            self.df_data, self.df_target = self.get_data()

    def get_dir(self, subdir=None):
        base = os.path.join(self.base_output_path,self.id)
        if subdir is None:
            return base
        if subdir is Subdirs.CV:
            return os.path.join(self.base_output_path, subdir.value)
        return os.path.join(base, subdir.value)

    def check_dirs(self, raise_error=True):
        if os.path.exists(self.base_output_path):
            os.makedirs(self.get_dir(), exist_ok=True)
            for subdirs in Subdirs:
                os.makedirs(self.get_dir(subdirs), exist_ok=True)

        else:
            error_str = 'output path must exist, please create the folder: "' + self.base_output_path + '"'
            if raise_error:
                raise UnboundLocalError(error_str)
            print(error_str)

    def backup_content(self):
        basepath = self.get_dir()
        if not os.path.exists(basepath):
            return

        path_content = os.listdir(basepath)
        if not path_content:
            return

        now = datetime.datetime.now()
        stardate = now.strftime("%Y%m%d%H%M%S")

        new_folder_path = os.path.join(basepath, stardate)
        os.makedirs(new_folder_path)

        for item in path_content:
            item_path = os.path.join(basepath, item)
            new_item_path = os.path.join(new_folder_path, item)
            try:
                datetime.datetime.strptime(item, "%Y%m%d%H%M%S")
            except ValueError:
                shutil.move(item_path, new_item_path)

    def clear_content(self):
        for dir in Subdirs:
            shutil.rmtree(self.get_dir(dir))

    def to_dict(self):
        ret = self.__dict__.copy()
        ret['df_data'] = None
        ret['df_target'] = None
        return ret

    def reset_flags(self):
        self.split_subsets_done = False
        self.pipeline_fitting_done = False
        self.models_fitting_done = False
        self.models_prediction_done = False

    @staticmethod
    def from_dict(dados):
        instancia = Experiment.__new__(Experiment)
        instancia.__dict__.update(dados)
        return instancia

    def get_data(self):
        df = pd.read_csv(self.input_file, sep=',')
        df_data = df[self.input_columns]
        df_target = df[self.output_column]
        return df_data, df_target

    def get_n_folds(self):
        return self.n_folds*self.n_repeats

    def get_param_pack_list(self):
        combinations = list(itertools.product(*self.model_grid.values()))
        pack_list = []
        for c_id, combination in enumerate(combinations):
            pack = dict(zip(self.model_grid.keys(), combination))
            pack_list.append(pack)
        return pack_list

    #Cross Validation

    def get_cv_filename(self, ifold):
        cv_name ='fold_{:d}_of_{:d}.pkl'.format(
                                        ifold,
                                        self.get_n_folds())
        return os.path.join(self.get_dir(Subdirs.CV),cv_name)

    def split_subsets(self, force=False, shuffle=default_shuffle, random_state=default_random_state):
        self.check_dirs()

        if self.split_subsets_done and not force:
            return

        # df_data, df_target = self.get_data()
        self.check_data()

        indexes = np.array(range(self.df_data.shape[0]))

        if self.test_subset != 0:
            x_subset, x_test, y_subset, _ = sklmodel.train_test_split(indexes,
                                                        self.df_target,
                                                        test_size=self.test_subset,
                                                        shuffle=shuffle,
                                                        stratify=self.df_target,
                                                        random_state=random_state)
        else:
            x_subset = indexes
            x_test = None
            y_subset = self.df_target

        if self.n_repeats == 1:
            cv = sklmodel.StratifiedKFold(n_splits=self.n_folds,
                            shuffle=shuffle,
                            random_state=random_state)
        else:
            cv = sklmodel.RepeatedStratifiedKFold(n_splits=self.n_folds,
                                     n_repeats=self.n_repeats,
                                     random_state=random_state)

        for ifold, (trn_id, val_id) in enumerate(cv.split(self.df_data.loc[x_subset,:], y_subset)):

            filename = self.get_cv_filename(ifold)
            if os.path.exists(filename):
                continue

            with open(filename,'wb') as file_handler:
                pickle.dump([x_subset[trn_id], x_subset[val_id], x_test],file_handler)

        self.split_subsets_done = True

    def get_subset_ids(self, ifold):
        filename = self.get_cv_filename(ifold)

        if not os.path.exists(filename):
            self.split_subsets(force=True)

        with open(filename,'rb') as file_handler:
            [trn_id, val_id, test_id] = pickle.load(file_handler)
        return trn_id, val_id, test_id

    #Pipeline

    def get_pipeline_filename(self, ifold):
        name ='fold_{:d}_of_{:d}.joblib'.format(
                                        ifold,
                                        self.get_n_folds())
        return os.path.join(self.get_dir(Subdirs.PIPELINE), name)

    def fit_pipeline(self, ifold):

        filename = self.get_pipeline_filename(ifold)
        if os.path.exists(filename):
            return

        # df_data, df_target = self.get_data()
        self.check_data()

        trn_id, val_id, test_id = self.get_subset_ids(ifold)

        if self.pipeline == 'StandardScaler':
            pipe = Pipeline(steps=[("scaler", StandardScaler())])
            pipe.fit(self.df_data.iloc[trn_id,:])
        elif self.pipeline == 'MinMaxScaler':
            pipe = Pipeline(steps=[("scaler", MinMaxScaler())])
            pipe.fit(self.df_data.iloc[trn_id, :])
        elif self.pipeline == 'BoxCox':
            pipe = Pipeline(steps=[
                ("preprocess", FunctionTransformer(_maximum)),
                ("scaler", PowerTransformer(method='box-cox'))
            ])
            pipe.fit(self.df_data.iloc[trn_id, :])
        elif self.pipeline == None:
            pipe = Pipeline(steps=[('passthrough', 'passthrough')])
        else:
            raise NotImplementedError("pipeline " + self.pipeline) 

        with open(filename,'wb') as file_handler:
            joblib.dump(pipe, file_handler)

        self.split_subsets_done = True

    def fit_pipelines(self, force=False, only_first_fold=False):
        self.check_dirs()

        if self.pipeline_fitting_done and not force:
            return

        for ifold in range(self.get_n_folds()):
            self.fit_pipeline(ifold)
            
            if only_first_fold:
                return

    def get_pipeline(self, ifold):
        filename = self.get_pipeline_filename(ifold)

        if not os.path.exists(filename):
            self.fit_pipeline(ifold)

        with open(filename,'rb') as file_handler:
            pipe = joblib.load(file_handler)
        return pipe

    #Fit model

    def get_model_filename(self, ifold, param_pack):
        name ='fold_{:d}_of_{:d}'.format(
                                        ifold,
                                        self.get_n_folds())
        for key, value in param_pack.items():
            name = name + '_' + key + '(' + str(value) + ")"

        return os.path.join(self.get_dir(Subdirs.MODEL), name + ".pkl")

    def fit_model(self, ifold, param_pack):
        filename = self.get_model_filename(ifold, param_pack)

        if os.path.exists(filename):
            return

        # df_data, df_target = self.get_data()
        self.check_data()

        trn_id, val_id, test_id = self.get_subset_ids(ifold)
        pipe = self.get_pipeline(ifold)

        trans_data = pd.DataFrame(pipe.transform(self.df_data), columns = self.df_data.columns, index = np.array(self.df_data.index))

        all_empty = all(value == '' for value in param_pack.values())
        if all_empty:
            constructor_str = self.model_constructor
        else:
            constructor_str = self.model_constructor.format(*param_pack.values())
        model = eval(constructor_str)

        model.fit(trans_data.iloc[trn_id],
                    self.df_target.iloc[trn_id],
                    val_X = trans_data.iloc[val_id],
                    val_Y = self.df_target.iloc[val_id])

        predictions = model.predict(trans_data)

        model.save(filename)

    def fit_models(self, only_first_fold=False, force=False):
        self.check_dirs()

        if self.models_fitting_done and not force:
            return

        for ifold in range(self.get_n_folds()):
            for param_pack in self.get_param_pack_list():
                self.fit_model(ifold, param_pack)

            if only_first_fold:
                return

        self.models_fitting_done = True

    def get_model(self, ifold, param_pack):
        filename = self.get_model_filename(ifold, param_pack)

        if not os.path.exists(filename):
            self.fit_model(ifold, param_pack)

        return ml.Base.load(filename)

    #Predict model

    def get_predict_filename(self, ifold, param_pack):
        name ='fold_{:d}_of_{:d}'.format(
                                        ifold,
                                        self.get_n_folds())
        for key, value in param_pack.items():
            name = name + '_' + key + '(' + str(value) + ")"

        return os.path.join(self.get_dir(Subdirs.PREDICTION), name + ".csv")

    def predict_model(self, ifold, param_pack):
        filename = self.get_predict_filename(ifold, param_pack)

        if os.path.exists(filename):
            return

        # df_data, df_target = self.get_data()
        self.check_data()
        pipe = self.get_pipeline(ifold)
        model = self.get_model(ifold, param_pack)

        trans_data = pd.DataFrame(pipe.transform(self.df_data), columns = self.df_data.columns, index = np.array(self.df_data.index))

        predictions = model.predict(trans_data)
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()

        df_predictions = pd.DataFrame({
            'predictions': predictions,
            'targets': self.df_target[self.df_target.columns[0]]
        })
        df_predictions.to_csv(filename, index=False)

    def predict_models(self, only_first_fold=False):
        self.check_dirs()

        if self.models_prediction_done:
            return

        for ifold in range(self.get_n_folds()):
            for param_pack in self.get_param_pack_list():
                self.predict_model(ifold, param_pack)
            
            if only_first_fold:
                return

        self.models_prediction_done = True

    def get_prediction(self, ifold, param_pack, only_if_ready=False):
        filename = self.get_predict_filename(ifold, param_pack)

        if not os.path.exists(filename):
            if only_if_ready:
                return None, None
            self.predict_model(ifold, param_pack)

        df = pd.read_csv(filename, sep=',')
        predictions = df['predictions']
        targets = df['targets']
        return targets, predictions

    #Evaluate model

    def get_evaluation_filename(self, subset):
        name = self.id + "_" + subset
        return os.path.join(self.get_dir(Subdirs.SCORE), name + ".tex")

    def get_evaluation(self, export_results=False, decision_threshold = 0.5, only_if_ready=False):

        trn_result = metrics.Grid_compiler()
        val_result = metrics.Grid_compiler()
        test_result = metrics.Grid_compiler()

        cont = 0
        for ifold in range(self.get_n_folds()):

            trn_id, val_id, test_id = self.get_subset_ids(ifold)

            for param_pack in self.get_param_pack_list():
                target, predict = self.get_prediction(ifold, param_pack, only_if_ready=only_if_ready)
                if target is None:
                    continue

                trn_result.eval(target.iloc[trn_id], predict[trn_id], param_pack, self.get_model_filename(ifold, param_pack), cont, decision_threshold)
                val_result.eval(target.iloc[val_id], predict[val_id], param_pack, self.get_model_filename(ifold, param_pack), cont, decision_threshold)
                if test_id is not None:
                    test_result.eval(target.iloc[test_id], predict[test_id], param_pack, self.get_model_filename(ifold, param_pack), cont, decision_threshold)

            cont = cont + 1

        if export_results:
            trn_result.save_tex(self.get_evaluation_filename('trn'))
            val_result.save_tex(self.get_evaluation_filename('val'))
            if test_id is not None:
                test_result.save_tex(self.get_evaluation_filename('test'))

        return Experiment_results(trn_result, val_result, test_result)

    def get_best_model_prediction(self, metric, subset):
        result = self.get_evaluation()
        bests = result[subset].get_bests()
        ifold = bests[metric]['id']
        param = result[subset].get_best_param(metric)
        return self.get_prediction(ifold, param)

class Trainer():

    def __init__(self, control_file):
        self.control_file = control_file
        self.experiments = {}
        self._load()

    def _load(self):
        if os.path.exists(self.control_file):
            with open(self.control_file, 'r') as f:
                exps_dict = json.load(f)
            for exp in exps_dict['configs']:
                 experiment = Experiment(exp)
                 self.experiments[experiment.id] = experiment

    def _save(self):
        exps_dict = {'configs': []}
        for key, exp in self.experiments.items():
             exps_dict['configs'].append(exp.to_dict())
        with open(self.control_file, 'w') as f:
            json.dump(exps_dict, f, indent=4)

    def get_experiment_result(self, id):
        return self.experiments[id].get_evaluation()

    def get_experiment_partial_result(self, id):
        return self.experiments[id].get_evaluation(only_if_ready=True)

    def get_experiment(self, id):
        return self.experiments[id]

    def config_experiment(self, config):
        id = config['id']
        self.experiments[id] = Experiment(config)
        self._save()

    def run(self, id = None, reset_experiments=False, backup_old=True, only_first_fold=False, metrics=None):
        try :
            if id == None:
                if only_first_fold:
                    for id in self.experiments.keys():
                        self.run(id, only_first_fold)
                    return
                result_dict = {}
                for id in self.experiments.keys():
                    result_dict[id] = self.run(id, only_first_fold)
                return result_dict
            
            if reset_experiments:
                if backup_old:
                    self.experiments[id].backup_content()
                else:
                    self.experiments[id].clear_content()

            if reset_experiments:
                self.experiments[id].reset_flags()

            self.experiments[id].fit_models(only_first_fold)
            self.experiments[id].predict_models(only_first_fold)
            if only_first_fold:
                return
            return self.experiments[id].get_evaluation(export_results=True)

        except Exception as e:
            self._save()
            raise e

        # self.generate_cv_indexes(id)
        # self.generate_pipelines(id)
        # self.fit(id, only_first_fold)
        # self.export_scores(id, metrics)


if __name__ == "__main__":
    config_file = "config.json"
    config = {
            "id": "mlp",
            "input_file": "./results/inputs/training.csv",
            "input_columns": [
                "Hora",
                "SPL PMA",
                "SNR",
                "Tempo sozinho"
            ],
            "output_column": "Visivel",
            "test_subset": 0,
            "n_folds": 3,
            "n_repeats": 1,
            "pipeline": "StandardScaler",
            "model_constructor": "SVM(kernel=\"linear\", {:s})",
            "model_grid": {
                "Reg": [
                    "C=0.1",
                    "C=2",
                    "nu=0.7"
                ]
            },
            "base_output_path": "./results",
        }

    trainer = Trainer(config_file)
    trainer.config_experiment(config)

    trainer.run(config['id'], reset_experiments=True, backup_old=False)

    # print('--- Result for 1 fold ---')
    # result = trainer.get_experiment_partial_result(config['id'])
    # for subset in Subsets:
    #     print(result[subset])
    # print('--- Result for all folds ---')
    # result = trainer.get_experiment_result(config['id'])
    # for subset in Subsets:
    #     print(result[subset])
