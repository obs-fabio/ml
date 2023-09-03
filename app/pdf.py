import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import entropy
import skimage.io as skimage

def get_files(directory: str, extension: str):
	file_list = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.endswith(extension):
				file_list.append(os.path.join(root, file))
	return sorted(file_list)

files = get_files("/home/sonar/Data/4classes/analysis/melgram","tiff")

final_files = []
for i in range(5):
	class_files = []
	for file in files:
		if f"classe{i}" in file:
			class_files.append(file)
	if class_files:
		final_files.append(class_files)

class_list = ['A', 'B', 'C', 'D']
class_pdf = []
class_xs = []
class_datas = []
min = math.inf
max = -math.inf
for f, files in enumerate(final_files):
	data = None
	for file in files:

		lofar_default = skimage.imread(file)
		
		if data is None:
			data = lofar_default
		else:
			data = np.concatenate((data, lofar_default), axis=1)

	xs = []
	pdfs = []
	datas = []
	medians = []
	for dimension in range(1,data.shape[0]-1):
		min = np.min([np.min(data[dimension, :]), min])
		max = np.max([np.max(data[dimension, :]), max])

		xs.append(np.linspace(np.min(data[dimension, :]), np.max(data[dimension, :]), 100))
		pdfs.append(gaussian_kde(data[dimension, :]))
		medians.append(np.median(data[dimension, :]))
		datas.append(data[dimension, :])
	
	for i in np.argsort(np.array(medians))[-3:]:
		y1 = pdfs[i](xs[i])
		plt.plot(xs[i], y1, label=f'PDF Vetor {i}')
	plt.ylabel('Densidade de Probabilidade')
	plt.legend()
	plt.title('PDF Estimada')
	plt.savefig(f"/home/sonar/Programming/Doutorado/lps_libs/ml/result/plots/class_{class_list[f]}.png")
	plt.close()

	class_pdf.append(pdfs)
	class_xs.append(xs)
	class_datas.append(datas)

# import itertools
# for i, j in list(itertools.combinations([0, 1, 2, 3], 2)):

# 	kl_divergence = []
# 	for k in range(len(class_pdf[i])):
# 		kl_div = entropy(class_pdf[i][k](class_xs[i][k]), class_pdf[j][k](class_xs[i][k]))
# 		kl_divergence.append(kl_div)

# 	plt.bar(range(len(kl_divergence)), kl_divergence)
# 	plt.xlabel('Dimensão')
# 	plt.ylabel('Divergência KL')
# 	plt.title('Divergência KL para Cada Dimensão')
# 	plt.savefig(f"/home/sonar/Programming/Doutorado/lps_libs/ml/result/plots/kl_{i}_{j}.png")
# 	plt.close()

import itertools
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

x = np.linspace(min, max, 1000)

for i, j in list(itertools.combinations([0, 1, 2, 3], 2)):

    kl_divergence = []
    mutual_information = []
    wasserstein_dist = []

    for k in range(len(class_pdf[i])):
        kl_div = entropy(class_pdf[i][k](x), class_pdf[j][k](x))
        kl_divergence.append(kl_div)

        mi = mutual_info_score(class_pdf[i][k](x), class_pdf[j][k](x))
        mutual_information.append(mi)

        wass_dist = wasserstein_distance(class_pdf[i][k](x), class_pdf[j][k](x))
        wasserstein_dist.append(wass_dist)

    plt.bar(range(len(kl_divergence)), kl_divergence)
    plt.xlabel('Dimensão')
    plt.ylabel('Divergência KL')
    plt.title('Divergência KL para Cada Dimensão')
    plt.savefig(f"/home/sonar/Programming/Doutorado/lps_libs/ml/result/plots/kl_{i}_{j}.png")
    plt.close()

    plt.bar(range(len(mutual_information)), mutual_information)
    plt.xlabel('Dimensão')
    plt.ylabel('Informação Mútua')
    plt.title('Informação Mútua para Cada Dimensão')
    plt.savefig(f"/home/sonar/Programming/Doutorado/lps_libs/ml/result/plots/mi_{i}_{j}.png")
    plt.close()

    plt.bar(range(len(wasserstein_dist)), wasserstein_dist)
    plt.xlabel('Dimensão')
    plt.ylabel('Distância de Wasserstein')
    plt.title('Distância de Wasserstein para Cada Dimensão')
    plt.savefig(f"/home/sonar/Programming/Doutorado/lps_libs/ml/result/plots/wass_{i}_{j}.png")
    plt.close()
