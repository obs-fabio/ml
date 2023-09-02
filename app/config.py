import os
import enum
import random
import numpy as np
import torch

import labsonar_ml.data_loader as ml_data

class Training(enum.Enum):
    PLOTS=0,
    GAN=1,
    DCGAN=2,
    DIFUSSION=3,
    CLASSIFIER_MLP_REAL=4,
    CLASSIFIER_MLP_SYNTHETIC=5,
    CLASSIFIER_MLP_JOINT=6,
    CLASSIFIER_CNN_REAL=7,
    CLASSIFIER_CNN_SYNTHETIC=8,
    CLASSIFIER_CNN_JOINT=9,
    AE_REAL=10,
    AE_SYNTHETIC=11,

    def __str__(self) -> str:
        if (self == Training.CLASSIFIER_MLP_REAL) or (self == Training.CLASSIFIER_CNN_REAL) or \
            (self == Training.CLASSIFIER_MLP_SYNTHETIC) or (self == Training.CLASSIFIER_CNN_SYNTHETIC) or \
            (self == Training.CLASSIFIER_MLP_JOINT) or (self == Training.CLASSIFIER_CNN_JOINT):
            return 'real'
        return self.name.lower()

class Artifacts(enum.Enum):
    MODEL=0,
    OUTPUT=1,

base_path = "./"
data_dir = 'data/4classes'
result_dir = 'result'

def get_data_dir():
    return os.path.join(base_path, data_dir)

def get_result_dir(i_fold, training: Training, artifact: Artifacts = None):
    if training == Training.PLOTS or artifact is None:
        return os.path.join(
            base_path,
            result_dir,
            *training.name.lower().split("_")
        )

    return os.path.join(
        base_path,
        result_dir,
        *training.name.lower().split("_"),
        *artifact.name.lower().split("_"),
        "fold_" + str(i_fold),
    )

def make_dirs():
    for i_fold in range(10):
        for training in Training:
            for artifact in Artifacts:
                os.makedirs(get_result_dir(i_fold, training, artifact), exist_ok=True)

def get_dataset_loro():
    return ml_data.init_four_classes_dataset(get_data_dir()).get_loro()

def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    for training in Training:
        print(get_result_dir(0, training))
        print(get_result_dir(0, training, Artifacts.MODEL))

    make_dirs()
