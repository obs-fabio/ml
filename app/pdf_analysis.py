import os
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
from scipy.special import kl_div
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
one_fold_only = False
one_class_only = False
plot_all = False

skip_folds = []
skip_class = []

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

class_means = {}
bins_means = {}

for class_id in ['A', 'B', 'C', 'D']:

	if class_id in skip_class:
		continue

	fold_kl_divergence = []
	fold_kl_gan = []
	fold_kl_ganspe = []
	fold_data_real = []
	fold_data_gan = []
	fold_data_ganspe = []

	for i_fold, (train_dataset, val_dataset, test_dataset) in enumerate(config.get_dataset_loro()):

		if i_fold in skip_folds:
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

		real_data_mean = np.mean(real_data, axis=1)
		gan_data_mean = np.mean(gan_data, axis=1)
		ganspe_data_mean = np.mean(ganspe_data, axis=1)

		kl_divergence = []
		kl_gan = []
		kl_ganspe = []
		kl_x = []
		for bin in range(1,real_data.shape[0]-5):
			kl_x.append(bin)

			size = np.min([real_data.shape[-1], gan_data.shape[-1], ganspe_data.shape[-1]])

			x = np.linspace(0, 1, 200)
			real = real_data[bin, :size]
			gan = gan_data[bin, :size]
			ganspe = ganspe_data[bin, :size]

			real = np.where(real == 0, 1e-10, real)
			gan = np.where(gan == 0, 1e-10, gan)
			ganspe = np.where(ganspe == 0, 1e-10, ganspe)

			min_g = np.min([np.min(real_data), np.min(gan_data), np.min(ganspe_data)])
			max_g = np.max([np.max(real_data), np.max(gan_data), np.max(ganspe_data)])
			bins = np.linspace(min_g, max_g, 50)

			real_bin_value, real_bin_edge, real_bin_patchs = plt.hist(real, bins=bins, density=True)
			if plot_all:
				plt.savefig(os.path.join(output_dir,f'pdf_{class_id}_{i_fold}_{bin}_real.png'))
			# plt.close()
			
			gan_bin_value, gan_bin_edge, gan_bin_patchs = plt.hist(gan, bins=bins, density=True)
			if plot_all:
				plt.savefig(os.path.join(output_dir,f'pdf_{class_id}_{i_fold}_{bin}_gan.png'))
			# plt.close()
			
			ganspe_bin_value, ganspe_bin_edge, ganspe_bin_patchs = plt.hist(ganspe, bins=bins, density=True)
			# if plot_all:
			# 	plt.savefig(os.path.join(output_dir,f'pdf_{class_id}_{i_fold}_{bin}_ganspe.png'))
			plt.close()

			# if plot_all:
			# 	plt.hist(real, bins=bins, density=True, color='blue', alpha=0.2)
			# 	plt.hist(gan, bins=bins, density=True, color='red', alpha=0.2)
			# 	plt.hist(ganspe, bins=bins, density=True, color='green', alpha=0.2)
			# 	plt.savefig(os.path.join(output_dir,f'pdf_{class_id}_{i_fold}_{bin}.png'))
			# 	plt.close()

			def kl_divergence_func(p, q):
				result = np.zeros_like(p)
				for i, value in enumerate(p):
					if ((p[i] == 0) or (q[i] == 0)):
						continue
					else:
						result[i] = p[i] * np.log(p[i] / q[i])
				return result

			gan_kl_div = kl_divergence_func(real_bin_value, gan_bin_value)
			ganspe_kl_div = kl_divergence_func(real_bin_value, ganspe_bin_value)

			if plot_all:
				plt.plot(bins[1:], gan_kl_div, label = "gan")
				plt.plot(bins[1:], ganspe_kl_div, label = "ganspe")
				plt.legend()
				plt.savefig(os.path.join(output_dir,f'kl_{class_id}_{i_fold}_{bin}.png'))
				plt.close()


				fig = plt.figure()
				ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
				ax2 = plt.subplot2grid((3, 1), (2, 0))

				ax1.hist(real, bins=bins, density=True, color='blue', alpha=0.3, label="real")
				ax1.hist(gan, bins=bins, density=True, color='red', alpha=0.3, label="gan")
				ax1.hist(ganspe, bins=bins, density=True, color='green', alpha=0.3, label="ganspe")
				ax1.legend()
				ax2.plot(bins[1:], gan_kl_div, label = "gan")
				ax2.plot(bins[1:], ganspe_kl_div, label = "ganspe")
				ax2.legend()
				plt.savefig(os.path.join(output_dir,f'pdf2_{class_id}_{i_fold}_{bin}.png'))
				plt.close()

			# gan_kl_div = kl_div(real, gan).sum()
			# ganspe_kl_div = kl_div(real, ganspe).sum()


			kl_divergence.append(np.sum(ganspe_kl_div)/np.sum(gan_kl_div))
			kl_gan.append(np.sum(gan_kl_div))
			kl_ganspe.append(np.sum(ganspe_kl_div))

		kl_divergence = np.array(kl_divergence)
		kl_gan = np.array(kl_gan)
		kl_ganspe = np.array(kl_ganspe)

		fig = plt.figure()
		ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
		ax2 = plt.subplot2grid((4, 1), (2, 0))
		ax3 = plt.subplot2grid((4, 1), (3, 0))

		x = range(128)

		ax1.plot(kl_x, real_data_mean[kl_x], color='blue', label="real")
		ax1.plot(kl_x, gan_data_mean[kl_x], color='red', label="gan")
		ax1.plot(kl_x, ganspe_data_mean[kl_x], color='green', label="ganspe")
		ax1.legend()

		kl_gan_sum_spe = np.sum(kl_gan[bin_selections[class_id]])
		kl_ganspe_sum_spe = np.sum(kl_ganspe[bin_selections[class_id]])
		kl_factor_spe = kl_ganspe_sum_spe/kl_gan_sum_spe
		
		kl_gan_sum = np.sum(kl_gan)
		kl_ganspe_sum = np.sum(kl_ganspe)
		kl_factor = kl_ganspe_sum/kl_gan_sum

		ax2.plot(kl_x, kl_gan, label = "gan(%1.3f)"%(kl_gan_sum))
		ax2.plot(kl_x, kl_ganspe, label = "ganspe(%1.3f)"%(kl_ganspe_sum))
		ax2.legend()

		ax3.plot(kl_x, kl_divergence, label = "norm")
		ax3.axhline(1, color='red', linestyle='--')
		ax3.set_ylim([0,2])
		ax3.legend()

		y1 = [np.min(real_data_mean[kl_x]),np.max(real_data_mean[kl_x])]
		y2 = [np.min(kl_gan),np.max(kl_gan)]
		y3 = [np.min(kl_divergence),np.max(kl_divergence)]

		for selected_range in selections[class_id]:
			ax1.fill_betweenx(y1, selected_range[0], selected_range[-1], color='red', alpha=0.2)
			ax2.fill_betweenx(y2, selected_range[0], selected_range[-1], color='red', alpha=0.2)
			ax3.fill_betweenx(y3, selected_range[0], selected_range[-1], color='red', alpha=0.2)

		ax1.set_title(f"{kl_factor} -> {kl_factor_spe}")
		fig.tight_layout()
		plt.savefig(os.path.join(output_dir,f'pdf3_{class_id}_{i_fold}.png'))
		plt.close()

		fold_kl_divergence.append(kl_divergence)
		fold_kl_gan.append(kl_gan)
		fold_kl_ganspe.append(kl_ganspe)
		fold_data_real.append(real_data_mean)
		fold_data_gan.append(gan_data_mean)
		fold_data_ganspe.append(ganspe_data_mean)


		if one_fold_only:
			break

	fold_kl_divergence.append(kl_divergence)
	fold_kl_gan.append(kl_gan)
	fold_kl_ganspe.append(kl_ganspe)
	fold_data_real.append(real_data_mean)
	fold_data_gan.append(gan_data_mean)
	fold_data_ganspe.append(ganspe_data_mean)

	fold_kl_divergence = np.array(fold_kl_divergence)
	fold_kl_gan = np.array(fold_kl_gan)
	fold_kl_ganspe = np.array(fold_kl_ganspe)
	fold_data_real = np.array(fold_data_real)
	fold_data_gan = np.array(fold_data_gan)
	fold_data_ganspe = np.array(fold_data_ganspe)

	fold_kl_divergence = np.mean(fold_kl_divergence, axis=0)
	fold_kl_gan = np.mean(fold_kl_gan, axis=0)
	fold_kl_ganspe = np.mean(fold_kl_ganspe, axis=0)
	fold_data_real = np.mean(fold_data_real, axis=0)
	fold_data_gan = np.mean(fold_data_gan, axis=0)
	fold_data_ganspe = np.mean(fold_data_ganspe, axis=0)

	fig = plt.figure()
	ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
	ax2 = plt.subplot2grid((4, 1), (2, 0))
	ax3 = plt.subplot2grid((4, 1), (3, 0))

	x = range(128)

	ax1.plot(kl_x, fold_data_real[kl_x], color='blue', label="real")
	ax1.plot(kl_x, fold_data_gan[kl_x], color='red', label="gan")
	ax1.plot(kl_x, fold_data_ganspe[kl_x], color='green', label="ganspe")
	ax1.legend()

	kl_gan_sum_spe = np.sum(fold_kl_gan[bin_selections[class_id]])
	kl_ganspe_sum_spe = np.sum(fold_kl_ganspe[bin_selections[class_id]])
	kl_factor_spe = kl_ganspe_sum_spe/kl_gan_sum_spe
	
	kl_gan_sum = np.sum(fold_kl_gan)
	kl_ganspe_sum = np.sum(fold_kl_ganspe)
	kl_factor = kl_ganspe_sum/kl_gan_sum

	ax2.plot(kl_x, fold_kl_gan, label = "gan(%1.3f)"%(kl_gan_sum))
	ax2.plot(kl_x, fold_kl_ganspe, label = "ganspe(%1.3f)"%(kl_ganspe_sum))
	ax2.legend()

	ax3.plot(kl_x, kl_divergence, label = "norm")
	ax3.axhline(1, color='red', linestyle='--')
	ax3.set_ylim([0,2])
	ax3.legend()

	y1 = [np.min(fold_data_real[kl_x]),np.max(fold_data_real[kl_x])]
	y2 = [np.min(fold_kl_gan),np.max(fold_kl_gan)]
	y3 = [np.min(fold_kl_divergence),np.max(fold_kl_divergence)]

	for selected_range in selections[class_id]:
		ax1.fill_betweenx(y1, selected_range[0], selected_range[-1], color='red', alpha=0.2)
		ax2.fill_betweenx(y2, selected_range[0], selected_range[-1], color='red', alpha=0.2)
		ax3.fill_betweenx(y3, selected_range[0], selected_range[-1], color='red', alpha=0.2)

	ax1.set_title(f"{kl_factor} -> {kl_factor_spe}")
	fig.tight_layout()
	plt.savefig(os.path.join(output_dir,f'pdf4_{class_id}.png'))
	plt.close()

	if one_class_only:
		break

	# print("Class ", class_id)
	# print("\tall: ", class_means[class_id])
	# print("\tbins: ", bins_means[class_id])

# print(class_means)






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
