from abc import ABC, abstractmethod
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tikz

from sklearn.utils.class_weight import compute_class_weight

class Base(ABC):
    def __init__(self):
        self.name = type(self).__name__
        self.model = None

    def __str__(self):
        return ("not " if (self.model == None) else "") + "fitted " + self.name + " model"

    @abstractmethod
    def fit(self, X, Y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        pass

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def get_class_weight(self,Y):
        if isinstance(Y, pd.core.frame.DataFrame):
            Y = Y.squeeze()
        weights = compute_class_weight('balanced', classes = np.unique(Y), y = list(Y))
        class_weight = {0: weights[0], 1: weights[1]}
        # class_weight = {0: 1.2, 1: 0.8}
        return class_weight

    def plot_predict_hist(self, X, Y, filepath=None, **kwargs):

        Y = Y.values
        predictions = self.predict(X)

        errors = np.abs(np.squeeze(predictions) - Y)

        positive_errors = errors[Y == 1]
        negative_errors = errors[Y == 0]

        plt.figure(figsize=(10, 7))
        plt.hist(positive_errors, bins=30, alpha=0.5, color='b', label='Positive Class')
        plt.hist(negative_errors, bins=30, alpha=0.5, color='r', label='Negative Class')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.legend()

        if filepath is not None:
            filename, extension = os.path.splitext(filepath)
            print(extension)
            if extension == ".tex":
                tikz.save(filepath)
            else:
                plt.savefig(filepath)
        else:
            plt.title('Prediction Error Histogram')
            plt.show()

    def eval_robustness(self, X, Y, eval_id, metric, decision_threshold=0.5):

        relevance = np.zeros(X.shape[1],)

        base_score = metric.eval(X.iloc[eval_id], Y[eval_id], decision_threshold)
        for iinput in range(X.shape[1]):

            buffer_data = np.copy(X)
            buffer_data[:,iinput] = np.mean(buffer_data[:,iinput])
            predictions = self.predict(buffer_data[eval_id])

            score = metric.Metric.eval_scores(predictions,Y[eval_id], decision_threshold)[str(metric)]

            relevance[iinput] = (score-base_score)/base_score
