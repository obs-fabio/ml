import os
import datetime
import shutil
import numpy as np
import torch
import torchvision
import torch.utils.data as torch_data

import labsonar_ml.data_loader as ml_data


def get_available_device():
	return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_available_device():
	if torch.cuda.is_available():
		device = torch.cuda.current_device()
		print(f"Using GPU: {torch.cuda.get_device_name(device)}")
	else:
		print("No GPU available, using CPU.")

def has_files(directory: str) -> bool:
	for root, _, files in os.walk(directory):
		for file in files:
			if not os.path.isdir(file):
				return True
	return False

def get_files(directory: str, extension: str):
	file_list = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.endswith(extension):
				file_list.append(os.path.join(root, file))
	return sorted(file_list)

def images_to_vectors(images):
	return images.view(images.size(0), -1)

def vectors_to_images(vectors, image_dim):
	return vectors.view(vectors.size(0), *image_dim)

def make_targets(n_samples: int, target: int, device):
	return torch.autograd.variable.Variable(torch.ones(n_samples, 1) * target).to(device)

def prepare_train_dir(basepath: str, backup=True):

	if backup:
		os.makedirs(basepath, exist_ok=True)

		if has_files(basepath):
			path_content = os.listdir(basepath)
			now = datetime.datetime.now()
			stardate = now.strftime("%Y%m%d%H%M%S")

			new_folder_path = os.path.join(basepath, stardate)
			os.makedirs(new_folder_path)

			for item in path_content:
				item_path = os.path.join(basepath, item)
				try:
					datetime.datetime.strptime(item, "%Y%m%d%H%M%S")
				except ValueError:
					new_item_path = os.path.join(new_folder_path, item)
					shutil.move(item_path, new_item_path)
	else:
		if os.path.exists(basepath):
			shutil.rmtree(basepath)
		os.makedirs(basepath, exist_ok=True)

def get_mnist_dataset_as_specialist(datapath: str = "data/", specialist_class_number: int = 1):
	transform = torchvision.transforms.Compose([
									torchvision.transforms.Resize(32, antialias=True),
									torchvision.transforms.ToTensor(),
									torchvision.transforms.Normalize((0.5,), (0.5,))])
	
	train = torchvision.datasets.MNIST(root=datapath, train=True, download=True, transform=transform)
	
	targets_idx = train.targets.detach().clone() == specialist_class_number

	return torch_data.dataset.Subset(train, np.where(targets_idx)[0])

def read_images(files, transform = None):
    images = []
    for file in files:
        image = ml_data.read_image(file, transform)
        image = image.view(-1)
        images.append(image.tolist())
    return np.array(images)
