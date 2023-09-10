import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.manifold as sk_manifold
import sklearn.decomposition as sk_dec


def export_tsne(data: np.ndarray, labels: np.ndarray, filename: str, **kwargs):
    title = kwargs.get('title', 't-SNE')
    xlabel = kwargs.get('xlabel', 'Dimension 1')
    ylabel = kwargs.get('ylabel', 'Dimension 2')

    tsne = sk_manifold.TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, max(20,len(unique_labels))))
    for i, label in enumerate(unique_labels):
        plt.scatter(tsne_data[labels == label, 0], tsne_data[labels == label, 1], color=colors[i], label=str(label))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def export_pca(data: np.ndarray, labels: np.ndarray, filename: str, **kwargs):
    title = kwargs.get('title', 'PCA')
    xlabel = kwargs.get('xlabel', 'Dimension 1')
    ylabel = kwargs.get('ylabel', 'Dimension 2')

    pca = sk_dec.PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, max(20,len(unique_labels))))
    for i, label in enumerate(unique_labels):
        plt.scatter(pca_data[labels == label, 0], pca_data[labels == label, 1], color=colors[i], label=str(label))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def export_kernel_pca(data: np.ndarray, labels: np.ndarray, filename: str, **kwargs):
    title = kwargs.get('title', 'Kernel PCA')
    xlabel = kwargs.get('xlabel', 'Dimension 1')
    ylabel = kwargs.get('ylabel', 'Dimension 2')

    kwargs['n_components'] = 2
    kernel_pca = sk_dec.KernelPCA(**kwargs)
    kernel_pca_data = kernel_pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, max(20,len(unique_labels))))
    for i, label in enumerate(unique_labels):
        plt.scatter(kernel_pca_data[labels == label, 0], kernel_pca_data[labels == label, 1], color=colors[i], label=str(label))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def export(data: np.ndarray, labels: np.ndarray, output_dir: str, id: str = ""):
    export_tsne(data, labels, os.path.join(output_dir,id + "t-sne.png"))
    export_pca(data, labels, os.path.join(output_dir, id + "pca.png"))


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
