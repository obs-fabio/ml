import torch
import pickle
import typing

class Serializable_model:
    def save(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str) -> 'Serializable_model':
        with open(file_path, "rb") as f:
            return pickle.load(f)


class Base(torch.nn.Module, Serializable_model):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__

    def __str__(self) -> str:
        return self.name

