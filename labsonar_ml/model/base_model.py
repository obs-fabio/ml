import torch
import pickle

class Serializable():
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

class Base(torch.nn.Module, Serializable):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__


