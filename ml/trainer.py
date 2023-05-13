import os, json
import hashlib
import pickle
import pandas as pd
import numpy as np
import re
import joblib
import itertools
import time
from abc import ABC, abstractmethod

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ml.models.base as ml
from ml.models.mlp import MLP
from ml.metrics.metrics import get_scores

def get_files(directory, extension):
	file_list = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.endswith(extension):
				file_list.append(os.path.join(root, file))
	return file_list


class Base_trainer(ABC):
    def __init__(self, control_file):
        self.control_file = control_file
        self.hash_map = {}
        self.trainings = {'configs':[]}
        self.load()

    def load(self):
        if os.path.exists(self.control_file):
            with open(self.control_file, 'r') as f:
                self.trainings = json.load(f)
        self.update_indexes()

    def save(self):
        with open(self.control_file, 'w') as f:
            json.dump(self.trainings, f, indent=4)

    def update_indexes(self):
        self.hash_map.clear()
        for i, c in enumerate(self.trainings['configs']):
            self.hash_map[c['hash_id']] = i

    @staticmethod
    def hash(str_id):
        return int(hashlib.md5(str_id.encode()).hexdigest(), 16)
    
    def get_data(self, hash_id):
        df = pd.read_csv(self.trainings['configs'][self.hash_map[hash_id]]['data_file'], sep=',')
        output_index = df.columns.get_loc(self.trainings['configs'][self.hash_map[hash_id]]['last_input_column']) + 1

        df_data = df[df.columns[:output_index]]
        df_target = df[[self.trainings['configs'][self.hash_map[hash_id]]['output_column']]]

        return df_data, df_target


class Trainer(Base_trainer):

    def __init__(self, control_file):
        super().__init__(control_file)

    def update_config(self, config):
        if config['hash_id'] in self.hash_map:
                self.trainings['configs'][self.hash_map[config['hash_id']]] = config
        else:
            self.hash_map[config['hash_id']] = len(self.trainings['configs'])
            self.trainings['configs'].append(config)
        super().save()

    def generate_cv_indexes(self, hash_id = None):
        if hash_id == None:
            for key, value in self.hash_map.items():
                self.generate_cv_indexes(key)
            return

        config = self.trainings['configs'][self.hash_map[hash_id]]

        if config['ids']:
            return

        os.makedirs(config['cross_validation_file_path'], exist_ok=True)

        cv = StratifiedKFold(n_splits=config['n_folds'],
                            shuffle=True,
                            random_state=42)

        df_data, df_target = self.get_data(hash_id)

        for ifold, (trn_id, val_id) in enumerate(cv.split(df_data,df_target)):
            cv_name ='%s_cross_validation_fold_%i_of_%i.pkl'%(
                            hash_id,
                            ifold,
                            config['n_folds'])

            with open(os.path.join(config['cross_validation_file_path'],cv_name),'wb') as file_handler:
                pickle.dump([trn_id, val_id],file_handler)

        self.trainings['configs'][self.hash_map[hash_id]]['ids'] = True
        super().save()

    def generate_pipelines(self, hash_id = None):
        if hash_id == None:
            for key, value in self.hash_map.items():
                self.generate_pipelines(key)
            return

        config = self.trainings['configs'][self.hash_map[hash_id]]
        
        if config['pipelines']:
            return
        
        os.makedirs(config['pipeline_file_path'], exist_ok=True)

        df_data, df_target = self.get_data(hash_id)

        files = get_files(config['cross_validation_file_path'], '.pkl')
        for filename in files:
            with open(filename,'rb') as file_handler:
                [trn_id, val_id] = pickle.load(file_handler)

            if config['pipeline'] == 'StandardScaler':
                pipe = Pipeline(steps=[("scaler", StandardScaler())])
                pipe.fit(df_data.iloc[trn_id,:])

            elif config['pipeline'] == None:
                pipe = Pipeline(steps=['passthrough', 'passthrough'])

            else:
                raise NotImplementedError("pipeline " + config['pipeline']) 

            path, rel_filename = os.path.split(filename)
            pipe_name = re.sub("cross_validation", "pipeline", rel_filename)

            with open(os.path.join(config['pipeline_file_path'],pipe_name),'wb') as file_handler:
                joblib.dump(pipe, file_handler)

        self.trainings['configs'][self.hash_map[hash_id]]['pipelines'] = True
        super().save()

    def fit(self, hash_id = None):
        if hash_id == None:
            for key, value in self.hash_map.items():
                self.fit(key)
            return
        
        config = self.trainings['configs'][self.hash_map[hash_id]]
        
        if config['fit']:
            return
        
        model_constructor = config['constructor']
        cv_file_path = config['cross_validation_file_path']
        pipe_file_path = config['pipeline_file_path']
        model_path = config['model_path']
        evaluate_path = config['evaluate_path']

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(evaluate_path, exist_ok=True)

        params = config['constructor_params']
        keys = params.keys()
        values = params.values()

        combinations = list(itertools.product(*values))

        df_data, df_target = self.get_data(hash_id)

        for combination in combinations:
            parameter_dict = dict(zip(keys, combination))

            evaluate_name = '%s_evaluate'%(hash_id)
            for key, value in parameter_dict.items():
                evaluate_name = evaluate_name + '_' + key + '_' + str(value)
            evaluate_name = evaluate_name + '.pkl'

            n_folds = config['n_folds']
            for ifold in range(n_folds):

                cv_name = '%s_cross_validation_fold_%i_of_%i.pkl'%(hash_id, ifold, n_folds)
                pipe_name = '%s_pipeline_fold_%i_of_%i.pkl'%(hash_id, ifold, n_folds)

                model_name = '%s_model'%(hash_id)
                for key, value in parameter_dict.items():
                    model_name = model_name + '_' + key + '_' + str(value)
                model_name = model_name + '_fold_%i_of_%i.pkl'%(ifold, n_folds)

                evaluate_name = '%s_evaluate'%(hash_id)
                for key, value in parameter_dict.items():
                    evaluate_name = evaluate_name + '_' + key + '_' + str(value)
                evaluate_name = evaluate_name + '_fold_%i_of_%i.pkl'%(ifold, n_folds)

                model_filename = os.path.join(model_path, model_name)
                evaluate_filename = os.path.join(evaluate_path, evaluate_name)

                if os.path.exists(model_filename) and os.path.exists(evaluate_filename):
                    continue

                with open(os.path.join(cv_file_path,cv_name),'rb') as file_handler:
                    [trn_id, val_id] = pickle.load(file_handler)

                with open(os.path.join(pipe_file_path,pipe_name),'rb') as file_handler:
                    pipe = joblib.load(file_handler)

                trans_data = pipe.transform(df_data)

                model = eval(model_constructor.format(*combination))

                start_time = time.time()
                model.fit(trans_data[trn_id,:],
                          df_target.iloc[trn_id,:],
                          val_X = trans_data[val_id, :],
                          val_Y = df_target.iloc[val_id, :])
                end_time = time.time()

                model.save(model_filename)

                trn_predictions = model.predict(trans_data[trn_id, :])
                val_predictions = model.predict(trans_data[val_id, :])
                all_predictions = model.predict(trans_data)

                trn_scores = get_scores(df_target.iloc[trn_id, :], trn_predictions)
                val_scores = get_scores(df_target.iloc[val_id, :], val_predictions)
                all_scores = get_scores(df_target.iloc[:,:], all_predictions)

                score = {
                    'trn_scores': trn_scores,
                    'val_scores': val_scores,
                    'all_scores': all_scores,
                    'trn_time': end_time-start_time
                }

                print(evaluate_filename)
                with open(evaluate_filename, 'w') as f:
                    json.dump(score, f, indent=4)


        self.trainings['configs'][self.hash_map[hash_id]]['fit'] = True
        self.trainings['configs'][self.hash_map[hash_id]]['evaluate'] = True
        super().save()


