import os, json
import hashlib
import pickle
import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ml.models.base as ml

def get_files(directory, extension):
	file_list = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.endswith(extension):
				file_list.append(os.path.join(root, file))
	return file_list


class Trainer():
    def __init__(self, control_file):
        self.control_file = control_file
        self.hash_map = {}
        self.trainings = {'configs':[]}
        self.__load()

    def __load(self):
        if os.path.exists(self.control_file):
            with open(self.control_file, 'r') as f:
                self.trainings = json.load(f)
        self.__update_indexes()

    def __save(self):
        with open(self.control_file, 'w') as f:
            json.dump(self.trainings, f, indent=4)
        self.__update_indexes()

    def __update_indexes(self):
        self.hash_map.clear()
        for i, c in enumerate(self.trainings['configs']):
            self.hash_map[c['hash_id']] = i

    @staticmethod
    def hash(str_id):
        return int(hashlib.md5(str_id.encode()).hexdigest(), 16)

    def update_config(self, config):
        if config['hash_id'] in self.hash_map:
                self.trainings['configs'][self.hash_map[config['hash_id']]] = config
        else:
            self.trainings['configs'].append(config)
        self.__save()

    def __get_data(self, hash_id):
        df = pd.read_csv(self.trainings['configs'][self.hash_map[hash_id]]['data_file'], sep=',')
        output_index = df.columns.get_loc(self.trainings['configs'][self.hash_map[hash_id]]['last_input_column']) + 1

        df_data = df[df.columns[:output_index]]
        df_target = df[[self.trainings['configs'][self.hash_map[hash_id]]['output_column']]]

        return df_data, df_target

    def generate_cv_indexes(self, hash_id = None):
        if hash_id == None:
            for key, value in self.hash_map.items():
                self.generate_cv_indexes(key)
            return

        if self.trainings['configs'][self.hash_map[hash_id]]['ids']:
            return

        os.makedirs(self.trainings['configs'][self.hash_map[hash_id]]['cross_validation_file_path'], exist_ok=True)

        cv = StratifiedKFold(n_splits=self.trainings['configs'][self.hash_map[hash_id]]['n_folds'],
                            shuffle=True,
                            random_state=42)

        df_data, df_target = self.__get_data(hash_id)

        for ifold, (trn_id, val_id) in enumerate(cv.split(df_data,df_target)):
            cv_name ='%s_cross_validation_fold_%i_of_%i.pkl'%(
                            self.trainings['configs'][self.hash_map[hash_id]]['hash_id'],
                            ifold,
                            self.trainings['configs'][self.hash_map[hash_id]]['n_folds'])

            with open(os.path.join(self.trainings['configs'][self.hash_map[hash_id]]['cross_validation_file_path'],cv_name),'wb') as file_handler:
                pickle.dump([trn_id, val_id],file_handler)

        self.trainings['configs'][self.hash_map[hash_id]]['ids'] = True
        self.__save()

    def generate_pipelines(self, hash_id = None):
        if hash_id == None:
            for key, value in self.hash_map.items():
                self.generate_pipelines(key)
            return
        
        if self.trainings['configs'][self.hash_map[hash_id]]['pipelines']:
            return
        
        os.makedirs(self.trainings['configs'][self.hash_map[hash_id]]['pipeline_file_path'], exist_ok=True)

        df_data, df_target = self.__get_data(hash_id)

        files = get_files(self.trainings['configs'][self.hash_map[hash_id]]['cross_validation_file_path'], '.pkl')
        for filename in files:
            with open(filename,'rb') as file_handler:
                [trn_id, val_id] = pickle.load(file_handler)

            if self.trainings['configs'][self.hash_map[hash_id]]['pipeline'] == 'StandardScaler':
                pipe = Pipeline(steps=[("scaler", StandardScaler())])
                pipe.fit(df_data.iloc[trn_id,:])

            elif self.trainings['configs'][self.hash_map[hash_id]]['pipeline'] == None:
                pipe = Pipeline(steps=['passthrough', 'passthrough'])

            else:
                raise NotImplementedError("pipeline " + self.trainings['configs'][self.hash_map[hash_id]]['pipeline']) 

            path, rel_filename = os.path.split(filename)
            pipe_name = re.sub("cross_validation", "pipeline", rel_filename)

            with open(os.path.join(self.trainings['configs'][self.hash_map[hash_id]]['pipeline_file_path'],pipe_name),'wb') as file_handler:
                joblib.dump(pipe, file_handler)

