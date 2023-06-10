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

import ml.models.base as ml
from ml.models.mlp import MLP
from ml.models.svm import SVM, KC
from ml.models.forest import RandomForest
import ml.metrics.metrics as metrics

def get_files(directory, extension):
	file_list = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.endswith(extension):
				file_list.append(os.path.join(root, file))
	return file_list

def _maximum(X):
    return np.maximum(X, 1e-10)

class Subdirs(Enum):
    CV = "cv"
    PIPELINE = "pipeline"
    MODEL = "model"
    EVALUATE = "evaluate"
    PREDICTION = "predict"
    SCORE = "score"
    PLOT = "plot"

class Experiment():
    default_shuffle = True
    default_random_state = 42

    def __init__(self, config):
        self.update(config)
        self.split_subsets_done = False
        self.pipeline_fit_done = False
        self.models_fit_done = False

    def get_dir(self, subdir=None):
        base = os.path.join(self.output_path,self.id)
        if subdir is None:
            return base
        return os.path.join(base, subdir.value)

    def check_dirs(self, raise_error=True):
        if os.path.exists(self.output_path):
            os.makedirs(self.get_dir(), exist_ok=True)
            for subdirs in Subdirs:
                os.makedirs(self.get_dir(subdirs), exist_ok=True)

        else:
            error_str = 'output path must exist, please create the folder: "' + self.output_path + '"'
            if raise_error:
                raise UnboundLocalError(error_str)
            print(error_str)

    def update(self, config):
        self.hash_id = Trainer.hash(config['id'])
        self.id = config['id']
        self.description = ""
        self.data_file = config['data_file']
        self.input_column = config['input_column']
        self.output_column = config['output_column']
        self.test_subset = config['test_subset']
        self.n_folds = config['n_folds']
        self.n_repeats = config['n_repeats']
        self.pipeline = config['pipeline']
        self.constructor = config['constructor']
        self.constructor_params = config['constructor_params']
        self.output_path = config['output_path']

        self.check_dirs(False)

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(dados):
        instancia = Experiment.__new__(Experiment)
        instancia.__dict__.update(dados)
        return instancia

    def get_data(self):
        df = pd.read_csv(self.data_file, sep=',')
        df_data = df[self.input_column]
        df_target = df[self.output_column]
        return df_data, df_target

    def get_n_folds(self):
        return self.n_folds*self.n_repeats

    def get_param_pack_list(self):
        combinations = list(itertools.product(*self.constructor_params.values()))
        pack_list = []
        for c_id, combination in enumerate(combinations):
            pack = dict(zip(self.constructor_params.keys(), combination))
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

        df_data, df_target = self.get_data()

        indexes = np.array(range(df_data.shape[0]))

        x_subset, x_test, y_subset, _ = sklmodel.train_test_split(indexes,
                                                       df_target,
                                                       test_size=self.test_subset,
                                                       shuffle=shuffle,
                                                       stratify=df_target,
                                                       random_state=random_state)

        if self.n_repeats == 1:
            cv = sklmodel.StratifiedKFold(n_splits=self.n_folds,
                            shuffle=shuffle,
                            random_state=random_state)
        else:
            cv = sklmodel.RepeatedStratifiedKFold(n_splits=self.n_folds,
                                     n_repeats=self.n_repeats,
                                     random_state=random_state)

        for ifold, (trn_id, val_id) in enumerate(cv.split(df_data.loc[x_subset,:], y_subset)):

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
        name ='fold_{:d}_of_{:d}.pkl'.format(
                                        ifold,
                                        self.get_n_folds())
        return os.path.join(self.get_dir(Subdirs.PIPELINE), name)

    def fit_pipeline(self, ifold):

        filename = self.get_pipeline_filename(ifold)
        if os.path.exists(filename):
            return

        df_data, df_target = self.get_data()

        trn_id, val_id, test_id = self.get_subset_ids(ifold)

        if self.pipeline == 'StandardScaler':
            pipe = Pipeline(steps=[("scaler", StandardScaler())])
            pipe.fit(df_data.iloc[trn_id,:])
        elif self.pipeline == 'MinMaxScaler':
            pipe = Pipeline(steps=[("scaler", MinMaxScaler())])
            pipe.fit(df_data.iloc[trn_id, :])
        elif self.pipeline == 'BoxCox':
            pipe = Pipeline(steps=[
                ("preprocess", FunctionTransformer(_maximum)),
                ("scaler", PowerTransformer(method='box-cox'))
            ])
            pipe.fit(df_data.iloc[trn_id, :])
        else:
            raise NotImplementedError("pipeline " + self.pipeline) 

        with open(filename,'wb') as file_handler:
            joblib.dump(pipe, file_handler)

        self.split_subsets_done = True

    def fit_pipelines(self, force=False):
        self.check_dirs()

        if self.pipeline_fit_done and not force:
            return

        for ifold in range(self.get_n_folds()):
            self.fit_pipeline(ifold)

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

        df_data, df_target = self.get_data()

        trn_id, val_id, test_id = self.get_subset_ids(ifold)
        pipe = self.get_pipeline(ifold)

        trans_data = pd.DataFrame(pipe.transform(df_data), columns = df_data.columns, index = np.array(df_data.index))

        model = eval(self.constructor.format(*param_pack.values()))

        model.fit(trans_data.iloc[trn_id],
                    df_target.iloc[trn_id],
                    val_X = trans_data.iloc[val_id],
                    val_Y = df_target.iloc[val_id])

        model.save(filename)

    def fit_models(self, only_first_fold=False, force=False):
        self.check_dirs()

        if self.models_fit_done and not force:
            return

        for ifold in range(self.get_n_folds()):
            for param_pack in self.get_param_pack_list():
                self.fit_model(ifold, param_pack)

            if only_first_fold:
                return

        self.models_fit_done = True

    def get_model(self, ifold, param_pack):
        filename = self.get_model_filename(ifold, param_pack)

        if not os.path.exists(filename):
            self.fit_model(ifold, param_pack)

        return ml.Base.load(filename)

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

class Trainer():
    def __init__(self, control_file):
        self.control_file = control_file
        self.experiments = {}
        self._load()

    @staticmethod
    def hash(str_id):
        return int(hashlib.md5(str_id.encode()).hexdigest(), 16)

    def _load(self):
        if os.path.exists(self.control_file):
            with open(self.control_file, 'r') as f:
                exps_dict = json.load(f)
            for exp in exps_dict['configs']:
                 exp = Experiment.from_dict(exp)
                 self.experiments[exp.id] = exp

    def _save(self):
        exps_dict = {'configs': []}
        for key, exp in self.experiments.items():
             exps_dict['configs'].append(exp.to_dict())
        with open(self.control_file, 'w') as f:
            json.dump(exps_dict, f, indent=4)

    def get_experiment(self, id):
        return self.experiments[id]

    def config_experiment(self, config):
        id = config['id']
        if id in self.experiments:
             self.experiments[id].update(config)
        else:
            self.experiments[id] = Experiment(config)
        self._save()

    def all(self, id = None, reset_experiments=False, backup_old=True, only_first_fold=False, metrics=None):
        try :
            if id == None:
                for id in self.experiments.keys():
                    self.all(id, only_first_fold)
                return
            
            if reset_experiments:
                if backup_old:
                    self.experiments[id].backup_content()
                else:
                    self.experiments[id].clear_content()

            self.experiments[id].fit_models(only_first_fold)

        except:
            self._save()

        # self.generate_cv_indexes(id)
        # self.generate_pipelines(id)
        # self.fit(id, only_first_fold)
        # self.export_scores(id, metrics)

if __name__ == "__main__":
    config_file = "config.json"
    config = {
            "id": "ab",
            "data_file": "./results/inputs/training.csv",
            "input_column": [
                "Hora",
                "SPL PMA",
                "SNR",
                "Tempo sozinho"
            ],
            "output_column": "Visivel",
            "test_subset": 0.1,
            "n_folds": 3,
            "n_repeats": 2,
            "pipeline": "StandardScaler",
            "constructor": "SVM(kernel=\"linear\", {:s})",
            "constructor_params": {
                "Reg": [
                    "C=0.1",
                    "C=2",
                    "nu=0.7"
                ]
            },
            "output_path": "./results",
        }

    trainer = Trainer(config_file)
    trainer.config_experiment(config)

    trainer.all(config['id'])
    # experiment = trainer.get_experiment(config['id'])
    # print(experiment.get_subset_ids(4))
