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

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ml.models.base as ml
from ml.models.mlp import MLP
import ml.metrics.metrics as metrics

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

        df_data = df[self.trainings['configs'][self.hash_map[hash_id]]['input_column']]
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

        df_data, df_target = self.get_data(hash_id)

        indexes = np.array(range(df_data.shape[0]))

        x_train, x_test, y_train, _ = train_test_split(indexes, df_target, test_size=config['test_subset'], shuffle=True, stratify=df_target, random_state=42)

        cv = StratifiedKFold(n_splits=config['n_folds'],
                            shuffle=True,
                            random_state=42)


        for ifold, (trn_id, val_id) in enumerate(cv.split(df_data.loc[x_train,:], y_train)):
            cv_name ='%s_cross_validation_fold_%i_of_%i.pkl'%(
                            hash_id,
                            ifold,
                            config['n_folds'])

            with open(os.path.join(config['cross_validation_file_path'],cv_name),'wb') as file_handler:
                pickle.dump([x_train[trn_id], x_train[val_id], x_test],file_handler)

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
                [trn_id, val_id, _] = pickle.load(file_handler)

            if config['pipeline'] == 'StandardScaler':
                pipe = Pipeline(steps=[("scaler", StandardScaler())])
                pipe.fit(df_data.iloc[trn_id,:])

            elif config['pipeline'] == None:
                pipe = Pipeline(steps=[('passthrough', 'passthrough')])

            else:
                raise NotImplementedError("pipeline " + config['pipeline']) 

            path, rel_filename = os.path.split(filename)
            pipe_name = re.sub("cross_validation", "pipeline", rel_filename)

            with open(os.path.join(config['pipeline_file_path'],pipe_name),'wb') as file_handler:
                joblib.dump(pipe, file_handler)

        self.trainings['configs'][self.hash_map[hash_id]]['pipelines'] = True
        super().save()

    def fit(self, hash_id = None, test_mode=False):
        if hash_id == None:
            for key, value in self.hash_map.items():
                self.fit(key, test_mode)
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

            n_folds = config['n_folds']
            for ifold in range(n_folds if not test_mode else 1):

                cv_name = '%s_cross_validation_fold_%i_of_%i.pkl'%(hash_id, ifold, n_folds)
                pipe_name = '%s_pipeline_fold_%i_of_%i.pkl'%(hash_id, ifold, n_folds)

                model_name = '%s_model'%(hash_id)
                for key, value in parameter_dict.items():
                    model_name = model_name + '_' + key + '_' + str(value)
                model_name = model_name + '_fold_%i_of_%i.pkl'%(ifold, n_folds)

                evaluate_name = '%s_evaluate'%(hash_id)
                for key, value in parameter_dict.items():
                    evaluate_name = evaluate_name + '_' + key + '_' + str(value)
                evaluate_name = evaluate_name + '_fold_%i_of_%i.json'%(ifold, n_folds)

                model_filename = os.path.join(model_path, model_name)
                evaluate_filename = os.path.join(evaluate_path, evaluate_name)

                if os.path.exists(model_filename) and os.path.exists(evaluate_filename) and not config['override']:
                    continue

                with open(os.path.join(cv_file_path,cv_name),'rb') as file_handler:
                    [trn_id, val_id, test_id] = pickle.load(file_handler)

                with open(os.path.join(pipe_file_path,pipe_name),'rb') as file_handler:
                    pipe = joblib.load(file_handler)

                trans_data = pipe.transform(df_data)

                if isinstance(trans_data, pd.DataFrame):
                    trans_data = trans_data.values

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
                test_predictions = model.predict(trans_data[test_id, :])
                all_predictions = model.predict(trans_data)

                trn_scores = metrics.get_scores(df_target.values[trn_id, :], trn_predictions)
                val_scores = metrics.get_scores(df_target.values[val_id, :], val_predictions)
                test_scores = metrics.get_scores(df_target.values[test_id, :], test_predictions)
                all_scores = metrics.get_scores(df_target.values, all_predictions)

                score = {
                    'trn_scores': trn_scores,
                    'val_scores': val_scores,
                    'test_scores': test_scores,
                    'all_scores': all_scores,
                    'trn_time': end_time-start_time
                }

                with open(evaluate_filename, 'w') as f:
                    json.dump(score, f, indent=4)


        self.trainings['configs'][self.hash_map[hash_id]]['fit'] = True
        self.trainings['configs'][self.hash_map[hash_id]]['evaluate'] = True
        super().save()

    def get_evaluation(self, hash_id = None):
        if hash_id == None:
            evaluations = {}
            for key, value in self.hash_map.items():
                evaluations[key] = self.get_evaluation(key)
            return evaluations
        
        config = self.trainings['configs'][self.hash_map[hash_id]]

        if not config['evaluate']:
            print(hash_id, ": not evaluated")
            raise UnboundLocalError(str(hash_id) +" it is not evaluated")

        model_path = config['model_path']
        evaluate_path = config['evaluate_path']

        params = config['constructor_params']
        keys = params.keys()
        values = params.values()

        combinations = list(itertools.product(*values))

        managers = {
            'training': metrics.Manager(),
            'validation': metrics.Manager(),
            'test': metrics.Manager(),
            'all': metrics.Manager()
        }

        for combination in combinations:
            parameter_dict = dict(zip(keys, combination))

            n_folds = config['n_folds']
            for ifold in range(n_folds):


                model_name = '%s_model'%(hash_id)
                for key, value in parameter_dict.items():
                    model_name = model_name + '_' + key + '_' + str(value)
                model_name = model_name + '_fold_%i_of_%i.pkl'%(ifold, n_folds)

                evaluate_name = '%s_evaluate'%(hash_id)
                for key, value in parameter_dict.items():
                    evaluate_name = evaluate_name + '_' + key + '_' + str(value)
                evaluate_name = evaluate_name + '_fold_%i_of_%i.json'%(ifold, n_folds)

                model_filename = os.path.join(model_path, model_name)
                evaluate_filename = os.path.join(evaluate_path, evaluate_name)

                if not os.path.exists(evaluate_filename):
                    continue

                with open(evaluate_filename, 'r') as f:
                    scores = json.load(f)

                managers['training'].add_score(parameter_dict.copy(), scores['trn_scores'].copy(), model_filename)
                managers['validation'].add_score(parameter_dict.copy(), scores['val_scores'].copy(), model_filename)
                managers['test'].add_score(parameter_dict.copy(), scores['test_scores'].copy(), model_filename)
                managers['all'].add_score(parameter_dict.copy(), scores['all_scores'].copy(), model_filename)

        return managers

    def export_scores(self, hash_id = None, valid_scores = None):
        if hash_id == None:
            for key, value in self.hash_map.items():
                self.export_scores(key, valid_scores)
            return
        
        config = self.trainings['configs'][self.hash_map[hash_id]]

        os.makedirs(config['score_path'], exist_ok=True)

        metrics = self.get_evaluation(hash_id = hash_id)

        for key, value in metrics.items():
            value.export_score_tex(os.path.join(config['score_path'], '%s_%s_score.tex'%(config['id'], key)), valid_scores)

    def do_all(self, hash_id = None, valid_scores = None, test_mode=False):
        if hash_id == None:
            for key, value in self.hash_map.items():
                self.do_all(key, valid_scores, test_mode)
            return

        self.generate_cv_indexes(hash_id = hash_id)
        self.generate_pipelines(hash_id = hash_id)
        self.fit(hash_id = hash_id, test_mode = test_mode)
        self.export_scores(hash_id = hash_id, valid_scores = valid_scores)


