import os
import enum
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

class Artifacts(enum.Enum):
    MODEL=0,
    OUTPUT=1,

base_path = "/tf/ml"
data_dir = 'data/4classes'
result_dir = 'result'

def get_data_dir():
    return os.path.join(base_path, data_dir)

def get_result_dir(i_fold, training: Training, artifact: Artifacts = None):
    return os.path.join(
        base_path,
        result_dir,
        str(i_fold),
        *training.name.lower().split("_"),
        *[] if artifact is None else (artifact.name.lower().split("_"))
    )

def make_dirs():
    for i_fold in range(10):
        for training in Training:
            for artifact in Artifacts:
                os.makedirs(get_result_dir(i_fold, training, artifact), exist_ok=True)

def get_dataset_loro():
    return ml_data.init_four_classes_dataset(get_data_dir()).get_loro()

if __name__ == "__main__":
    for training in Training:
        print(get_result_dir(0, training))
        print(get_result_dir(0, training, Artifacts.MODEL))

    make_dirs()
