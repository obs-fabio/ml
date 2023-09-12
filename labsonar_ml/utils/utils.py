import os
import datetime
import shutil
import typing
import numpy as np
import random
import torch


def get_available_device() -> typing.Union[str, torch.device]:
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

def get_files(directory: str, extension: str) -> typing.List[str]:
	file_list = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.endswith(extension):
				file_list.append(os.path.join(root, file))
	return sorted(file_list)


def make_targets(n_samples: int, target: int, device: typing.Union[str, torch.device]) -> torch.Tensor:
	return torch.autograd.variable.Variable(torch.ones(n_samples, 1) * target).to(device)


def prepare_train_dir(basepath: str, backup: bool = True):

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


def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
