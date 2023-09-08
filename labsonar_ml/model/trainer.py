import typing
import tqdm
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import sklearn.utils.class_weight as sk_utils

import torch
import torch.utils.data as torch_data
import torchvision

import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.model.base_model as ml_model


def fit_specialist_classifier(model: ml_model.Base,
                              train_dataset: torchvision.dataset.Dataset,
                              validation_dataset: torchvision.dataset.Dataset,
                              class_id: int,
                              batch_size: int,
                              n_epochs: int,
                              learning_rate: float,
                              optimizer: typing.Optional[torch.optim.Optimizer] = None,
                              device: typing.Union[str, torch.device] = ml_utils.get_available_device()) -> typing.Tuple[typing.List[float], typing.List[float]]:
    """treina um modelo como classificador especialista

    Returns:
        typing.Tuple[typing.List[float], typing.List[float]]: erro ao longo do treinamento (no conjunto de treino e de validação)
    """

    train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch_data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    train_n_batchs = len(train_loader)
    validation_n_batchs = len(validation_loader)

    train_error = []
    val_error = []
    for _ in tqdm.tqdm(range(n_epochs), leave=False, desc="Epochs"):

        runningLoss = 0.0
        for batch, (samples, classes) in enumerate(train_loader):

            optimizer.zero_grad()

            targets = torch.where(classes == class_id, torch.tensor(1.0), torch.tensor(0.0)).to(device)
            targets_cpu = targets.cpu().numpy()

            samples = samples.to(device)
            predictions = model(samples)


            class_weights = sk_utils.compute_class_weight('balanced', classes=np.unique(targets_cpu), y=targets_cpu)
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            loss_func = torch.nn.BCELoss(weight=class_weights)

            loss = loss_func(predictions, targets)
            loss.backward()
            optimizer.step()

            runningLoss += loss.items()

        train_error.append(runningLoss/train_n_batchs)

        runningLoss = 0.0
        with torch.no_grad():
            for batch, (samples, classes) in enumerate(validation_loader):

                targets = torch.where(classes == class_id, torch.tensor(1.0), torch.tensor(0.0)).to(device)
                targets_cpu = targets.cpu().numpy()

                samples = samples.to(device)
                predictions = model(samples)

                class_weights = sk_utils.compute_class_weight('balanced', classes=np.unique(targets_cpu), y=targets_cpu)
                class_weights = torch.tensor(class_weights, dtype=torch.float)
                loss_func = torch.nn.BCELoss(weight=class_weights)

                loss = loss_func(predictions, targets)

            #batch_val_error_list.append(loss.item())
            runningLoss += loss.item()

        val_error.append(runningLoss/validation_n_batchs)

    return np.array(train_error), np.array(val_error)

@torch.no_grad()
def eval_specialist_classifier(model: ml_model.Base,
                              dataset: torchvision.dataset.Dataset,
                              class_id: int,
                              batch_size: int = 32,
                              device: typing.Union[str, torch.device] = ml_utils.get_available_device()):

    data_loader = torch_data.DataLoader(dataset, batch_size=batch_size)

    real = None
    predict = None
    for samples, classes in data_loader:

        targets = torch.where(classes == class_id, torch.tensor(1.0), torch.tensor(0.0)).to(device)
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
def eval_specialist_classifiers(classes: typing.List[str],
                                classifiers: typing.List[torch.nn.Model],
                                dataset: torchvision.dataset.Dataset,
                                batch_size: int = 32,
                                device: typing.Union[str, torch.device] = ml_utils.get_available_device(),) -> pd.DataFrame:

    reals = []
    predicts = []
    for id, _ in enumerate(classes):

        real, predict = eval_specialist_classifier(
            model = classifiers[id],
            class_id = id,
            dataset = dataset,
            batch_size = batch_size,
            device = device
        )

        reals.append(real)
        predicts.append(predict)

    data = np.array(reals + predicts)

    columns = [f'real_class_{class_id}' for class_id in classes] + \
            [f'predict_class_{class_id}' for class_id in classes]

    df = pd.DataFrame(data, columns).T
    return df

def eval_metrics(dataframe: pd.DataFrame,
                 class_list: typing.List[str]) -> typing.Tuple[typing.List[float], float, float, float]:
    """ avalia métrica com base num dataframe de predição do tipo da função eval_specialist_classifiers

    Returns:
        typing.Tuple[typing.List[float], float, float, float]: confusion_matrix, f1-score, recall, accuracy
    """
    n_classes = len(class_list)

    #separa colunas com targets e predições
    real_columns = dataframe.columns[:n_classes]
    predict_columns = dataframe.columns[n_classes:n_classes*2]

    #funções que mapeiam nome da coluna do dataframe em nome da classe
    real_to_index = {label: class_list[index] for index, label in enumerate(real_columns)}
    predict_to_index = {label: class_list[index] for index, label in enumerate(predict_columns)}

    #compila as n_classes colunas de targets e predições em nome da coluna com o maiore valor (winner-takes-all)
    dataframe['real'] = dataframe[real_columns].idxmax(axis=1)
    dataframe['predict'] = dataframe[predict_columns].idxmax(axis=1)

    #usa os mapas para converter os indices aos nomes das classes
    dataframe['real'] = dataframe['real'].map(real_to_index)
    dataframe['predict'] = dataframe['predict'].map(predict_to_index)

    cm = sk_metrics.confusion_matrix(dataframe['real'], dataframe['predict'], labels=class_list)

    scores = sk_metrics.classification_report(dataframe['real'], dataframe['predict'], labels=class_list, output_dict=True)
    scores = scores['weighted avg']

    accuracy = sk_metrics.accuracy_score(dataframe['real'], dataframe['predict'])

    return cm, scores['f1-score'], scores['recall'], accuracy