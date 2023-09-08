import os
import enum
import random
import typing
import numpy as np
import torch

import labsonar_ml.data_loader as ml_data

base_path = "./"
data_dir = 'data/4classes'
result_dir = 'result'

class Training(enum.Enum):
    PLOTS=0,
    GAN = 1,
    SPEC_GAN = 2,
    CLASSIFIER=3,

    def __str__(self) -> str:
        return self.name.lower()

class Artifacts(enum.Enum):
    MODEL=0,
    OUTPUT=1,

    def __str__(self) -> str:
        return self.name.lower()


def get_data_dir() -> str:
    return os.path.join(base_path, data_dir)

def get_result_dir(id_training: typing.List[str], artifact: Artifacts = None, i_fold: int = None) -> str:
    arg_list = [base_path, result_dir]

    if isinstance(id_training, list):
        for id in id_training:
            arg_list.append(str(id))
    else:
        arg_list.append(str(id_training))

    if artifact is not None:
        arg_list.append(str(artifact))

    if i_fold is not None:
        arg_list.append("fold_" + str(i_fold))

    return os.path.join(*arg_list)

def make_dirs():
    for i_fold in range(10):
        for training in Training:
            if training == Training.PLOTS:
                os.makedirs(get_result_dir(training), exist_ok=True)
            else:
                for artifact in Artifacts:
                    os.makedirs(get_result_dir(training, artifact, i_fold), exist_ok=True)

def get_dataset_loro():
    return ml_data.init_four_classes_dataset(get_data_dir()).get_loro()

def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

specialist_bin_selections = {
	'A': [range(2,9), range(15,23), range(31,39), range(71,99)],
	'B': [range(2,8), range(26,32), range(38,44), range(65,91)],
	'C': [range(1,8), range(14,19), range(28,35), range(70,75)],
	'D': [range(1,10), range(14,18), range(33,38), range(55,62)],
}

def get_specialist_selected_bins() -> typing.Dict:
    bin_selections = {}
    for id, list_index in specialist_bin_selections.items():
        bin_selections[id] = []
        for list in list_index:
            for i in list:
                bin_selections[id].append(i)

    return bin_selections

if __name__ == "__main__":
    make_dirs()
