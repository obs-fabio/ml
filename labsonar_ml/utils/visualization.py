import os
import string
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

# def export_tsne(data: np.ndarray, labels: np.ndarray, filename: str, **kwargs):
#     title = kwargs.get('title', 't-SNE')

#     tsne = TSNE(n_components=2, random_state=42)
#     tsne_data = tsne.fit_transform(data)

#     unique_labels = np.unique(labels)

#     plt.figure(figsize=(8, 6))

#     plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels.astype(float), cmap='viridis', s=10)
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#     plt.title(title)
#     plt.xlabel('Dimension 1')
#     plt.ylabel('Dimension 2')
#     plt.savefig(filename, bbox_inches='tight')

def export_tsne(data: np.ndarray, labels: np.ndarray, filename: str, **kwargs):
    title = kwargs.get('title', 't-SNE')

    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, max(20,len(unique_labels))))
    for i, label in enumerate(unique_labels):
        plt.scatter(tsne_data[labels == label, 0], tsne_data[labels == label, 1], color=colors[i], label=str(label))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(filename, bbox_inches='tight')

def export_pca(data: np.ndarray, labels: np.ndarray, filename: str, **kwargs):
    title = kwargs.get('title', 'PCA')

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        plt.scatter(pca_data[labels == label, 0], pca_data[labels == label, 1], color=colors[i], label=str(label))
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(filename)

def export_kernel_pca(data: np.ndarray, labels: np.ndarray, filename: str, **kwargs):
    title = kwargs.get('title', 'Kernel PCA')

    kwargs['n_components'] = 2
    kernel_pca = KernelPCA(**kwargs)
    kernel_pca_data = kernel_pca.fit_transform(data)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        plt.scatter(kernel_pca_data[labels == label, 0], kernel_pca_data[labels == label, 1], color=colors[i], label=str(label))
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('Kernel PCA Component 1')
    plt.ylabel('Kernel PCA Component 2')
    plt.savefig(filename)

def export(data: np.ndarray, labels: np.ndarray, output_dir: str, id: str = ""):
    export_tsne(data, labels, os.path.join(output_dir,id + "t-sne.png"))
    export_pca(data, labels, os.path.join(output_dir, id + "pca.png"))
    # export_kernel_pca(data, labels, os.path.join(output_dir, "poly_3_pca.png"), kernel='poly', degree=3)
    # export_kernel_pca(data, labels, os.path.join(output_dir, "poly_5_pca.png"), kernel='poly', degree=5)
    # export_kernel_pca(data, labels, os.path.join(output_dir, "rbf_pca.png"), kernel='rbf')


if __name__ == "__main__":

    output_dir = "./plots"
    for type in ['normalized', 'relative']:

        data = pd.read_csv('./data/{:s}_background.csv'.format(type))
        spectro = data.iloc[:, 4:].to_numpy()

        dir = os.path.join(output_dir,type)
        os.makedirs(dir, exist_ok=True)

        export_tsne(spectro, np.zeros((data.shape[0],)), filename=os.path.join(dir, "tse.png"))

        for i, column in enumerate(['Rain','Temperature','Wind']):
            label = data[column].to_numpy()
            export_tsne(spectro, label, filename=os.path.join(dir, column.lower() + "_tse.png"))

    # output_dir = "./plots"
    # data = pd.read_csv('./data/ambient_background.csv')
    # spectro = data.iloc[:, 4:].to_numpy()

    # export_tsne(spectro, np.zeros((data.shape[0],)), filename=os.path.join(output_dir, "tse.png"))

    # for i, column in enumerate(['Rain','Temperature','Wind']):
    #     label = data[column].to_numpy()
    #     export_tsne(spectro, label, filename=os.path.join(output_dir, column + "_tse.png"))

    # output_dir = "./plots"
    # data = pd.read_csv('./data/label_background.csv')
    # spectro = data.iloc[:, 4:].to_numpy()

    # export_tsne(spectro, np.zeros((data.shape[0],)), filename=os.path.join(output_dir, "tse.png"))

    # for i, column in enumerate(['Rain','Temperature','Wind']):
    #     label = data[column].to_numpy()
    #     export_tsne(spectro, label, filename=os.path.join(output_dir, column + "_tse.png"))

    # output_dir = "./plots"
    # data = pd.read_csv('./data/ambient_background.csv')
    # spectro = data.iloc[:, 4:].to_numpy()

    # # export(spectro, np.zeros((data.shape[0],)), output_dir)
    # for i, column in enumerate(['rain','temperature','wind']):
    #     aux = data.iloc[:, i + 1].to_numpy()
        
    #     filt = aux[np.where(aux != '---')].astype(float)
    #     filt = filt[np.where(filt != 0)]

    #     plt.hist(filt)
    #     plt.xlabel('{:s}'.format(column))
    #     plt.ylabel('Frequência')
    #     plt.title('Histograma {:s}'.format(column))
    #     plt.savefig(os.path.join(output_dir,'histogram_{:s}.png'.format(column)))
    #     plt.close()

    #     print(column)
    #     print("\t min: ", np.min(filt))
    #     print("\t max: ", np.max(filt))
    #     print("\t mean: ", np.mean(filt))
    #     print("\t median: ", np.median(filt))