import os, tqdm
import typing
import numpy as np
import matplotlib.pyplot as plt

import torchvision

import app.config as config
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import labsonar_ml.utils.visualization as ml_vis


def read_images(files, transform = None):
	images = None
	for file in sorted(files):
		image = ml_data.read_image(file, transform)

		if images is None:
			images = image.numpy()
		else:
			images = np.concatenate((images, image), axis=2)

	return images

def kl_divergence_func(p, q):

	p = np.where(p == 0, 1e-10, p)
	q = np.where(q == 0, 1e-10, q)

	result = np.zeros_like(p)
	for i, value in enumerate(p):
		result[i] = p[i] * np.log(p[i] / q[i])
	return result


def run(reset: bool = False,
		backup: bool = False,
		train: bool = True,
		evaluate: bool = True,
		one_fold_only: bool = True,
		one_class_only: bool = True,
		skip_folds: typing.List[int] = [],
		skip_class: typing.List[str] = [],
		):

	plot_bins = [30, 50, 75, 80, 85, 90, 95]

	training_dicts = []
	for sd_reg_factor in [1, 0.5, 0.001]:
		training_dicts.append({
				'dir': [config.Training.SPEC_GAN, f"{sd_reg_factor}"],
				'plot_dir': [config.Training.PLOTS, f"{sd_reg_factor}"],
				'file_id': str(config.Training.SPEC_GAN) + "_" + f"{sd_reg_factor}",
			})

	transform=torchvision.transforms.Compose([
			torchvision.transforms.Grayscale(),
			torchvision.transforms.ToTensor(),
		])
	
	selections = config.specialist_bin_selections
	bin_selections = config.get_specialist_selected_bins()


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
			real_data_mean = np.mean(real_data, axis=1)

			ref_model_dir = config.get_result_dir([config.Training.SPEC_GAN, f"{0}"],
										 artifact=config.Artifacts.OUTPUT,
										 i_fold=i_fold)

			gan_files = ml_utils.get_files(directory=os.path.join(ref_model_dir,class_id),
								  extension="png")

			gan_data = read_images(gan_files[:int(0.7*len(gan_files))], transform)
			gan_data = gan_data.reshape(gan_data.shape[-2], gan_data.shape[-1])
			gan_data_mean = np.mean(gan_data, axis=1)

			for training_dict in training_dicts:

				training_plot_dir = config.get_result_dir(training_dict['plot_dir'])
				bin_analysis_dir = os.path.join(training_plot_dir, "bin")
				os.makedirs(training_plot_dir, exist_ok=True)
				os.makedirs(bin_analysis_dir, exist_ok=True)


				ganspe_files = ml_utils.get_files(
												directory=os.path.join(
														config.get_result_dir(
																	training_dict['dir'],
																	artifact=config.Artifacts.OUTPUT,
																	i_fold=i_fold,),
														class_id),
												extension="png")


				ganspe_data = read_images(ganspe_files[:int(0.7*len(ganspe_files))], transform)
				ganspe_data = ganspe_data.reshape(ganspe_data.shape[-2], ganspe_data.shape[-1])
				ganspe_data_mean = np.mean(ganspe_data, axis=1)


				kl_divergence = []
				kl_gan = []
				kl_ganspe = []
				kl_x = []

				for bin in range(1,real_data.shape[0]-5):
					kl_x.append(bin)

					size = np.min([real_data.shape[-1], gan_data.shape[-1], ganspe_data.shape[-1]])

					real = real_data[bin, :size]
					gan = gan_data[bin, :size]
					ganspe = ganspe_data[bin, :size]

					min_g = np.min([np.min(real_data), np.min(gan_data), np.min(ganspe_data)])
					max_g = np.max([np.max(real_data), np.max(gan_data), np.max(ganspe_data)])
					bins = np.linspace(min_g, max_g, 200)

					real_bin_value, _, _ = plt.hist(real, bins=bins, density=True)
					plt.close()
					
					gan_bin_value, _, _ = plt.hist(gan, bins=bins, density=True)
					plt.close()
					
					ganspe_bin_value, _, _ = plt.hist(ganspe, bins=bins, density=True)
					plt.close()

					gan_kl_div = kl_divergence_func(real_bin_value, gan_bin_value)
					ganspe_kl_div = kl_divergence_func(real_bin_value, ganspe_bin_value)

					if bin in plot_bins:

						fig = plt.figure()
						ax1 = plt.subplot2grid((4, 1), (0, 0))
						ax2 = plt.subplot2grid((4, 1), (1, 0))
						ax3 = plt.subplot2grid((4, 1), (2, 0))
						ax4 = plt.subplot2grid((4, 1), (3, 0))

						ax1.hist(real, bins=bins, density=True, color='blue', alpha=0.3, label="real")
						ax2.hist(gan, bins=bins, density=True, color='red', alpha=0.3, label="gan")
						ax3.hist(ganspe, bins=bins, density=True, color='green', alpha=0.3, label="ganspe")
						ax1.legend(), ax2.legend(), ax3.legend()
						ax4.plot(bins[1:], gan_kl_div, label = "gan")
						ax4.plot(bins[1:], ganspe_kl_div, label = "ganspe")
						ax4.legend()
						fig.tight_layout()
						plt.savefig(os.path.join(bin_analysis_dir,f'bin_analysis_{class_id}_{i_fold}_{bin}.png'))
						plt.close()

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


				ax1.plot(kl_x, real_data_mean[kl_x], color='blue', label="real")
				ax1.plot(kl_x, gan_data_mean[kl_x], color='red', label="gan")
				ax1.plot(kl_x, ganspe_data_mean[kl_x], color='green', label="ganspe")
				ax1.legend()

				kl_gan_sum_spe = np.median(kl_gan[bin_selections[class_id]])
				kl_ganspe_sum_spe = np.median(kl_ganspe[bin_selections[class_id]])
				kl_factor_spe = kl_ganspe_sum_spe/kl_gan_sum_spe
				
				kl_gan_sum = np.median(kl_gan)
				kl_ganspe_sum = np.median(kl_ganspe)
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
				plt.savefig(os.path.join(training_plot_dir, f'kl_{class_id}_{i_fold}.png'))
				plt.close()

				fold_kl_divergence.append(kl_divergence)
				fold_kl_gan.append(kl_gan)
				fold_kl_ganspe.append(kl_ganspe)
				fold_data_real.append(real_data_mean)
				fold_data_gan.append(gan_data_mean)
				fold_data_ganspe.append(ganspe_data_mean)

			if one_fold_only:
				break

		if one_class_only:
			break


if __name__ == "__main__":
	ml_utils.print_available_device()
	run()