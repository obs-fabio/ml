from enum import Enum
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import itertools

def confusion_matrix(y_target, y_predict):
    tn, fp, fn, tp = cm(y_target, y_predict, normalize='true').ravel()
    return (tn, fp, fn, tp)

def sensitivity(y_target, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_target, y_predict)
    sensitivity = tp / (tp+fn)
    return sensitivity

def specificity(y_target, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_target, y_predict)
    specificity = tn / (tn+fp)
    return specificity

def auc(y_target,y_predict):
    return roc_auc_score(y_target, y_predict)

def sp_index(y_target, y_predict):
    spec = sensitivity(y_target, y_predict)
    sens = sensitivity(y_target, y_predict)
    return np.sqrt(spec*sens)*np.mean([spec, sens])

def acc(y_target, y_predict):
    return accuracy_score(y_target, y_predict)

class Scores(Enum):
    AUC = 1
    SENSITIVITY = 2
    SPECIFICITY = 3
    SP_INDEX = 4
    ACC = 5
    CONFUSION_MATRIX = 6

def get_scores(y_target, y_predict):
    return {
        Scores.AUC.value: auc(y_target, y_predict),
        Scores.SENSITIVITY.value: sensitivity(y_target, y_predict),
        Scores.SPECIFICITY.value: specificity(y_target, y_predict),
        Scores.SP_INDEX.value: sp_index(y_target, y_predict),
        Scores.ACC.value: acc(y_target, y_predict),
        Scores.CONFUSION_MATRIX.value: (confusion_matrix(y_target, y_predict)),
    }



