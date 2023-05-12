from abc import ABC, abstractmethod
import pickle

class Base(ABC):
    def __init__(self):
        self.name = type(self).__name__
        self.model = None

    def __str__(self):
        m_str = 'Class MLPModel\n'
        if self.trained:
            m_str += 'Model is fitted, '
        else:
            m_str += 'Model is not fitted, '
        m_str += 'instance created with %i hidden neurons'%(self.n_hidden_neurons) 
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