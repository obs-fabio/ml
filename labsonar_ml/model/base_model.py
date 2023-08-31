import torch
import pickle
import tqdm
import typing
import numpy as np
import pandas as pd
import torch.utils.data as torch_data

import labsonar_ml.data_loader as ml_data
import labsonar_ml.utils.utils as ml_utils

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


def fit_specialist_classifier(model, dataset, class_id, batch_size, n_epochs, lr, linearize_data):

    data_loader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.BCELoss()

    device = ml_utils.get_available_device()

    error_list = []
    for _ in tqdm.tqdm(range(n_epochs), leave=False, desc="Epochs"):

        for bacth, (samples, classes) in enumerate(data_loader):

            optimizer.zero_grad()

            targets = torch.where(classes == class_id, torch.tensor(1.0), torch.tensor(0.0)).to(device)

            if linearize_data:
                samples = torch.autograd.variable.Variable(ml_utils.images_to_vectors(samples)).to(device)
            else:
                samples = samples.to(device)

            outputs = model(samples)
            loss = loss_func(outputs.squeeze(1), targets)
            loss.backward()
            optimizer.step()

            error_list.append(loss.item())

    return np.array(error_list)

@torch.no_grad()
def eval_specialist_classifier(model, dataset, class_id, linearize_data):

    data_loader = torch_data.DataLoader(dataset, batch_size=32)
    device = ml_utils.get_available_device()

    real = None
    predict = None
    for samples, classes in data_loader:

        targets = torch.where(classes == class_id, torch.tensor(1.0), torch.tensor(0.0)).to(device)

        if linearize_data:
            samples = torch.autograd.variable.Variable(ml_utils.images_to_vectors(samples)).to(device)
        else:
            samples = samples.to(device)

        outputs = model(samples)

        if real is None:
            real = targets.cpu().numpy()
            predict = outputs.cpu().numpy()
        else:
            real = np.concatenate((real, targets.cpu().numpy()), axis=0)
            predict = np.concatenate((predict, outputs.cpu().numpy()), axis=0)

    return real.reshape(real.shape[0],), predict.reshape(predict.shape[0],)

@torch.no_grad()
def eval_specialist_classifiers(classes: typing.List[str], classifiers, dir: str, transform, linearize_data, datasets = None) -> pd.DataFrame:

    if datasets is None:
        return eval_specialist_classifiers(classes, classifiers, dir, transform, linearize_data, "train"), \
                eval_specialist_classifiers(classes, classifiers, dir, transform, linearize_data, "val"), \
                eval_specialist_classifiers(classes, classifiers, dir, transform, linearize_data, "test")

    reals = []
    predicts = []
    for id, class_id in enumerate(classes):

        syn_train_dataset, syn_val_dataset, syn_test_dataset = ml_data.load_synthetic_dataset(dir, transform)

        if datasets == "train":
            dataset = syn_train_dataset
        elif datasets == "val":
            dataset = syn_val_dataset
        else:
            dataset = syn_test_dataset

        real, predict = eval_specialist_classifier(
            model=classifiers[id],
            class_id=id,
            dataset=dataset,
            linearize_data=linearize_data
        )

        reals.append(real)
        predicts.append(predict)

    data = np.array(reals + predicts)

    columns = [f'real_class{class_id}' for class_id in classes] + \
            [f'predict_class{class_id}' for class_id in classes]

    df = pd.DataFrame(data, columns).T
    return df
