from abc import ABC, abstractmethod
import pickle
import numpy as np
import matplotlib.pyplot as plt
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
    def predict(self, X, output_as_classifier=True, **kwargs):
        pass

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def get_class_weight(self,Y):
        weights = compute_class_weight('balanced', classes = np.unique(Y), y = list(Y))
        class_weight = {0: weights[0], 1: weights[1]}
        return class_weight

    def plot_predict_hist(self, X, Y, save_file=None, **kwargs):

        Y = Y.values
        predictions = self.predict(X, output_as_classifier=False)

        errors = np.abs(np.squeeze(predictions) - Y)

        positive_errors = errors[Y == 1]
        negative_errors = errors[Y == 0]

        plt.figure(figsize=(10, 6))
        plt.plot(errors)
        plt.plot(positive_errors)
        plt.plot(negative_errors)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(positive_errors, bins=30, alpha=0.5, color='b', label='Positive Class')
        # plt.hist(negative_errors, bins=30, alpha=0.5, color='r', label='Negative Class')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Histogram')
        plt.legend()
        if save_file is None:
            plt.show()
        else:
            plt.savefig(save_file)

        plt.figure(figsize=(10, 6))
        # plt.hist(positive_errors, bins=30, alpha=0.5, color='b', label='Positive Class')
        plt.hist(negative_errors, bins=30, alpha=0.5, color='r', label='Negative Class')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Histogram')
        plt.legend()
        if save_file is None:
            plt.show()
        else:
            plt.savefig(save_file)