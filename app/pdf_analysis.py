import os
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
import skimage.io as skimage
from PIL import Image

import torchvision

import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import app.config as config



selections = {
	'A': [range(2,9), range(15,23), range(31,39), range(71,99)],
	'B': [range(2,8), range(26,32), range(38,44), range(65,91)],
	'C': [range(1,8), range(14,19), range(28,35), range(70,75)],
	'D': [range(1,10), range(14,18), range(33,38), range(55,62)],
}


bin_selections = {}
for id, list_index in selections.items():
	bin_selections[id] = []
	for list in list_index:
		for i in list:
			bin_selections[id].append(i)


reset=True
backup=True
one_fold_only = True
one_class_only = True

skip_folds = []
ship_class = []

transform=torchvision.transforms.Compose([
		torchvision.transforms.Grayscale(),
		torchvision.transforms.ToTensor(),
	])

def read_images(files, transform = None):
	images = None
	for file in sorted(files):
		image = ml_data.read_image(file, transform)

		if images is None:
			images = image.numpy()
		else:
			images = np.concatenate((images, image), axis=2)

	return images


output_dir = config.get_result_dir(0, config.Training.PLOTS)
output_dir = os.path.join(output_dir, "pdf")
os.makedirs(output_dir, exist_ok=True)

for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), leave=False):

	if i_fold in skip_folds:
		continue

	for class_id in train_dataset.get_classes():

		if one_fold_only and class_id in ship_class:
			continue

		real_data = read_images(train_dataset.get_files(class_id=class_id, run_id=None), transform)
		real_data = real_data.reshape(real_data.shape[-2], real_data.shape[-1])

		# real_data = real_data[75:95,:]
		# real_data = (real_data * 255).astype(np.uint8)
		# image = Image.fromarray(real_data, mode='L')
		# image.save(os.path.join(output_dir,f'{class_id}_{i_fold}.png'))

		gan_files = ml_utils.get_files(
										directory=os.path.join(
												config.get_result_dir(i_fold=i_fold,
															training=config.Training.GAN,
															artifact=config.Artifacts.OUTPUT),
												class_id),
										 extension="png")

		gan_data = read_images(gan_files[:int(0.7*len(gan_files))], transform)
		gan_data = gan_data.reshape(gan_data.shape[-2], gan_data.shape[-1])

		ganspe_files = ml_utils.get_files(
										directory=os.path.join(
												config.get_result_dir(i_fold=i_fold,
															training=config.Training.GANSPE,
															artifact=config.Artifacts.OUTPUT),
												class_id),
										 extension="png")


		ganspe_data = read_images(ganspe_files[:int(0.7*len(ganspe_files))], transform)
		ganspe_data = ganspe_data.reshape(ganspe_data.shape[-2], ganspe_data.shape[-1])


		kl_divergence = []
		for bin in range(1,real_data.shape[0]-1):

			try:
				x = np.linspace(0, 1, 500)

				real_pdf = sci.gaussian_kde(real_data[bin, :].reshape(-1))
				gan_pdf = sci.gaussian_kde(gan_data[bin, :].reshape(-1))
				ganspe_pdf = sci.gaussian_kde(ganspe_data[bin, :].reshape(-1))

				gan_kl_div = sci.entropy(real_pdf(x), gan_pdf(x))
				ganspe_kl_div = sci.entropy(real_pdf(x), ganspe_pdf(x))
				kl_divergence.append(ganspe_kl_div/gan_kl_div)

			except:
				kl_divergence.append(1)

			# print(gan_kl_div, "/", ganspe_kl_div)

		plt.plot(kl_divergence)
		plt.xlabel('Bin')
		plt.ylabel('Normalize KL divergence')
		plt.title(f'{class_id}_{i_fold}')
		plt.savefig(os.path.join(output_dir,f'{class_id}_{i_fold}.png'))
		plt.close()

		if one_class_only:
			break

	if one_fold_only:
		break










# def get_files(directory: str, extension: str):
# 	file_list = []
# 	for root, _, files in os.walk(directory):
# 		for file in files:
# 			if file.endswith(extension):
# 				file_list.append(os.path.join(root, file))
# 	return sorted(file_list)

# files = get_files("/home/sonar/Data/4classes/analysis/melgram","tiff")

# final_files = []
# for i in range(5):
# 	class_files = []
# 	for file in files:
# 		if f"classe{i}" in file:
# 			class_files.append(file)
# 	if class_files:
# 		final_files.append(class_files)

# class_list = ['A', 'B', 'C', 'D']
# class_pdf = []
# class_xs = []
# class_datas = []
# min = math.inf
# max = -math.inf
# for f, files in enumerate(final_files):
# 	data = None
# 	for file in files:

# 		lofar_default = skimage.imread(file)
		
# 		if data is None:
# 			data = lofar_default
# 		else:
# 			data = np.concatenate((data, lofar_default), axis=1)

# 	xs = []
# 	pdfs = []
# 	datas = []
# 	medians = []
# 	for dimension in range(1,data.shape[0]-1):
# 		min = np.min([np.min(data[dimension, :]), min])
# 		max = np.max([np.max(data[dimension, :]), max])

# 		xs.append(np.linspace(np.min(data[dimension, :]), np.max(data[dimension, :]), 100))
# 		pdfs.append(gaussian_kde(data[dimension, :]))
# 		medians.append(np.median(data[dimension, :]))
# 		datas.append(data[dimension, :])
	
# 	for i in np.argsort(np.array(medians))[-3:]:
# 		y1 = pdfs[i](xs[i])
# 		plt.plot(xs[i], y1, label=f'PDF Vetor {i}')
# 	plt.ylabel('Densidade de Probabilidade')
# 	plt.legend()
# 	plt.title('PDF Estimada')
# 	plt.savefig(f"/home/sonar/Programming/Doutorado/lps_libs/ml/result/plots/class_{class_list[f]}.png")
# 	plt.close()

# 	class_pdf.append(pdfs)
# 	class_xs.append(xs)
# 	class_datas.append(datas)

# # import itertools
# # for i, j in list(itertools.combinations([0, 1, 2, 3], 2)):

# # 	kl_divergence = []
# # 	for k in range(len(class_pdf[i])):
# # 		kl_div = entropy(class_pdf[i][k](class_xs[i][k]), class_pdf[j][k](class_xs[i][k]))
# # 		kl_divergence.append(kl_div)

# # 	plt.bar(range(len(kl_divergence)), kl_divergence)
# # 	plt.xlabel('Dimensão')
# # 	plt.ylabel('Divergência KL')
# # 	plt.title('Divergência KL para Cada Dimensão')
# # 	plt.savefig(f"/home/sonar/Programming/Doutorado/lps_libs/ml/result/plots/kl_{i}_{j}.png")
# # 	plt.close()

# import itertools
# from scipy.stats import entropy
# from sklearn.metrics import mutual_info_score
# from scipy.stats import wasserstein_distance
# import matplotlib.pyplot as plt

# x = np.linspace(min, max, 1000)

# for i, j in list(itertools.combinations([0, 1, 2, 3], 2)):

# 	kl_divergence = []
# 	mutual_information = []
# 	wasserstein_dist = []

# 	for k in range(len(class_pdf[i])):
# 		kl_div = entropy(class_pdf[i][k](x), class_pdf[j][k](x))
# 		kl_divergence.append(kl_div)

# 		mi = mutual_info_score(class_pdf[i][k](x), class_pdf[j][k](x))
# 		mutual_information.append(mi)

# 		wass_dist = wasserstein_distance(class_pdf[i][k](x), class_pdf[j][k](x))
# 		wasserstein_dist.append(wass_dist)

# 	plt.bar(range(len(kl_divergence)), kl_divergence)
# 	plt.xlabel('Dimensão')
# 	plt.ylabel('Divergência KL')
# 	plt.title('Divergência KL para Cada Dimensão')
# 	plt.savefig(f"/home/sonar/Programming/Doutorado/lps_libs/ml/result/plots/kl_{i}_{j}.png")
# 	plt.close()

# 	plt.bar(range(len(mutual_information)), mutual_information)
# 	plt.xlabel('Dimensão')
# 	plt.ylabel('Informação Mútua')
# 	plt.title('Informação Mútua para Cada Dimensão')
# 	plt.savefig(f"/home/sonar/Programming/Doutorado/lps_libs/ml/result/plots/mi_{i}_{j}.png")
# 	plt.close()

# 	plt.bar(range(len(wasserstein_dist)), wasserstein_dist)
# 	plt.xlabel('Dimensão')
# 	plt.ylabel('Distância de Wasserstein')
# 	plt.title('Distância de Wasserstein para Cada Dimensão')
# 	plt.savefig(f"/home/sonar/Programming/Doutorado/lps_libs/ml/result/plots/wass_{i}_{j}.png")
# 	plt.close()
