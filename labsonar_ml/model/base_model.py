import torch
import pickle
import tqdm
import typing
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import sklearn.utils.class_weight as sk_utils
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


def fit_specialist_classifier(model, trainset, validationset, class_id, batch_size, n_epochs, lr, linearize_data):

    train_loader = torch_data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validation_loader = torch_data.DataLoader(validationset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = ml_utils.get_available_device()
    model.to(device)

    batch_error_list = []
    batch_val_error_list = []
    error_list = []
    val_error_list = []
    for _ in tqdm.tqdm(range(n_epochs), leave=False, desc="Epochs"):

        runningLoss = 0.0

        for batch, (samples, classes) in enumerate(train_loader):

            optimizer.zero_grad()

            targets = torch.where(classes == class_id, torch.tensor(1.0), torch.tensor(0.0)).to(device)

            if linearize_data:
                samples = torch.autograd.variable.Variable(ml_utils.images_to_vectors(samples)).to(device)
            else:
                samples = samples.to(device)

            outputs = model(samples)

            y = targets.cpu().numpy()
            class_weights = sk_utils.compute_class_weight('balanced',classes=np.unique(y),y=y)
            class_weights = torch.tensor(class_weights,dtype=torch.float)

            loss_func = torch.nn.BCELoss()

            loss = loss_func(outputs.squeeze(1), targets)
            loss.backward()
            optimizer.step()
            

            runningLoss += loss.item()
            #batch_error_list.append(loss.item())
        
        error_list.append(runningLoss/batch)
        runningLoss = 0.0

        with torch.no_grad():
            for batch, (samples, classes) in enumerate(validation_loader):

                targets = torch.where(classes == class_id, torch.tensor(1.0), torch.tensor(0.0)).to(device)

                if linearize_data:
                    samples = torch.autograd.variable.Variable(ml_utils.images_to_vectors(samples)).to(device)
                else:
                    samples = samples.to(device)

                outputs = model(samples)
                loss = loss_func(outputs.squeeze(1), targets)

            #batch_val_error_list.append(loss.item())
            runningLoss += loss.item()

        val_error_list.append(runningLoss/batch)
        

    return np.array(error_list), np.array(val_error_list)

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
def eval_specialist_classifiers(classes: typing.List[str], classifiers, dataset, linearize_data, datasets = None) -> pd.DataFrame:

    reals = []
    predicts = []
    for id, class_id in enumerate(classes):

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

def eval_metrics(dataframe, class_list):

    real_columns = dataframe.columns[:4]
    predict_columns = dataframe.columns[4:8]

    real_to_index = {label: class_list[index] for index, label in enumerate(real_columns)}
    predict_to_index = {label: class_list[index] for index, label in enumerate(predict_columns)}

    dataframe['real'] = dataframe[real_columns].idxmax(axis=1)
    dataframe['predict'] = dataframe[predict_columns].idxmax(axis=1)
    dataframe['real'] = dataframe['real'].map(real_to_index)
    dataframe['predict'] = dataframe['predict'].map(predict_to_index)

    cm = sk_metrics.confusion_matrix(dataframe['real'], dataframe['predict'], labels=class_list)

    scores = sk_metrics.classification_report(dataframe['real'], dataframe['predict'], labels=class_list, output_dict=True)
    accuracy = sk_metrics.accuracy_score(dataframe['real'], dataframe['predict'])

    scores = scores['weighted avg']
    return cm, scores['f1-score'], scores['recall'], accuracy