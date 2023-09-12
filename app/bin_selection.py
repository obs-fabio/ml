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


def run(reset: bool = True,
		backup: bool = False,
		train: bool = True,
		evaluate: bool = True,
		one_fold_only: bool = False,
		one_class_only: bool = False,
		skip_folds: typing.List[int] = [],
		skip_class: typing.List[str] = [],
		):

	transform=torchvision.transforms.Compose([
			torchvision.transforms.Grayscale(),
			torchvision.transforms.ToTensor(),
		])

	selections = config.specialist_bin_selections
	bin_selections = config.get_specialist_selected_bins()

	result_dir = config.get_result_dir([config.Training.PLOTS, f"selection"])
	os.makedirs(result_dir, exist_ok=True)

	for i_fold, (train_dataset, val_dataset, test_dataset) in enumerate(config.get_dataset_loro()):

		for class_id in train_dataset.get_classes():

			if class_id in skip_class:
				continue

			train_files = train_dataset.get_files(class_id=class_id, run_id=None)
			val_files = val_dataset.get_files(class_id=class_id, run_id=None)
			test_files = test_dataset.get_files(class_id=class_id, run_id=None)
			files = train_files + val_files + test_files

			data = read_images(files, transform)
			data = data.reshape(data.shape[-2], data.shape[-1])
			data = np.mean(data, axis=1)


			fig = plt.figure()
			plt.plot(data, color='blue')

			y = [np.min(data),np.max(data)]

			for selected_range in selections[class_id]:
				plt.fill_betweenx(y, selected_range[0], selected_range[-1], color='red', alpha=0.2)

			fig.tight_layout()
			plt.savefig(os.path.join(result_dir, f'{class_id}.png'))
			plt.close()

		break


if __name__ == "__main__":
	ml_utils.print_available_device()
	run()