import os
import numpy as np
import typing
import PIL
import torch
import torchvision
import torch.utils.data as torch_data


class Base_dataset (torch_data.Dataset):

    def __init__(self,
                 samples: typing.List[typing.Tuple[str, str, str]],
                 classes: typing.List[str],
                 runs: typing.Dict[str, typing.List[str]],
                 transform: torchvision.transforms.Compose):
        super().__init__()
        self._samples = samples
        self._classes = sorted(classes)
        self._runs = runs
        self._transform: torchvision.transforms.Compose = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx) -> typing.Tuple[torch.Tensor, int]:
        image_path, class_id, _ = self._samples[idx]
        image = read_image(image_path, self._transform)
        return image, self._classes.index(class_id)

    def get_classes(self) -> typing.List[str]:
        return self._classes.copy()

    def get_runs(self, class_id: str) -> typing.List[str]:
        return list(self._runs[class_id])

    def get_files(self, class_id: str, run_id: str) -> typing.List[str]:
        files = []
        for image_path, _class_id, _run_id in self._samples:
            if (class_id == _class_id and ((run_id == _run_id) or run_id == None)) or class_id == None:
                files.append(image_path)
        return files

    def filt_dataset(self, class_id: str) -> 'Base_dataset':
        samples = []
        for image_path, _class_id, _run_id in self._samples:
            if class_id == _class_id:
                samples.append([image_path, _class_id, _run_id])

        return Base_dataset(samples, [class_id], self.get_runs(class_id), self._transform)

    def get_transform(self) -> torchvision.transforms.Compose:
        return self._transform.copy()


class Dataset_manager (Base_dataset):

    def __init__(self,
                 samples: typing.List[typing.Tuple[str, str, str]],
                 classes: typing.List[str],
                 runs: typing.Dict[str, typing.List[str]],
                 transform: torchvision.transforms.Compose):
        super().__init__(samples, classes, runs, transform)

    def get_loro(self) -> typing.List[typing.Tuple[typing.Type[Base_dataset], typing.Type[Base_dataset], typing.Type[Base_dataset]]]:

        max_runs = 0
        for class_id in self._classes:
            max_runs = max(max_runs, len(self._runs[class_id]))

        subsets = []
        for run_index in range(max_runs):
            test = []
            val = []
            train = []
            train_runs = {}
            val_runs = {}
            test_runs = {}

            for _class_id in self._classes:

                test_index = run_index % len(self._runs[_class_id])
                val_index = (run_index + 1) % len(self._runs[_class_id])

                train_runs[_class_id] = [run for index, run in enumerate(self._runs[_class_id])
                                         if index not in [test_index, val_index]]

                val_runs[_class_id] = [self._runs[_class_id][val_index]]

                test_runs[_class_id] = [self._runs[_class_id][test_index]]

            for image_path, _class_id, _run_id in self._samples:

                test_index = run_index % len(self._runs[_class_id])
                val_index = (run_index + 1) % len(self._runs[_class_id])

                if test_index == self._runs[_class_id].index(_run_id):
                    test.append([image_path, _class_id, _run_id])
                elif val_index == self._runs[_class_id].index(_run_id):
                    val.append([image_path, _class_id, _run_id])
                else:
                    train.append([image_path, _class_id, _run_id])

            subsets.append([Base_dataset(train, self._classes, train_runs, self._transform),
                            Base_dataset(val, self._classes, val_runs, self._transform),
                            Base_dataset(test, self._classes, test_runs, self._transform)])

        return subsets


def read_image(image_file, transform = None, linearize=False) -> PIL.Image:
    image = PIL.Image.open(image_file).convert('RGB')
    if transform:
        image = transform(image)
    if linearize:
        image = image.view(-1)
    return image

def read_images(files, transform = None, linearize=False):
	images = []
	for file in files:
		image = read_image(file, transform, linearize)
		images.append(image.tolist())
	return np.array(images)


def get_defaul_transform() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

def init_four_classes_dataset(base_dir: str,
        transform: torchvision.transforms.Compose = None,
        extension: str = '.png') -> typing.Type[Dataset_manager]:
    _samples = []
    _classes = []
    _runs = {}

    transform = transform if transform is not None else get_defaul_transform()

    for class_dir in sorted(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_dir)

        if os.path.isdir(class_path):
            class_id = class_dir

            for run_dir in sorted(os.listdir(class_path)):
                run_path = os.path.join(class_path, run_dir)

                if os.path.isdir(run_path):
                    run_id = run_dir

                    for image_file in os.listdir(run_path):
                        if image_file.endswith(extension):
                            image_path = os.path.join(run_path, image_file)
                            _samples.append((image_path, class_id, run_id))

                            if class_id not in _classes:
                                _classes.append(class_id)

                            if class_id not in _runs:
                                _runs[class_id] = []

                            if run_id not in _runs[class_id]:
                                _runs[class_id].append(run_id)

    return Dataset_manager(_samples, _classes, _runs, transform)

def load_synthetic_dataset(dir: str,
                           transform = None,
                           train_ratio: int = 0.7,
                           val_ratio: int = 0.15) -> typing.Tuple[torch_data.Dataset, torch_data.Dataset, torch_data.Dataset]:

    transform = transform if transform is not None else get_defaul_transform()

    syn_dataset = torchvision.datasets.ImageFolder(dir, transform)

    train_size = int(train_ratio * len(syn_dataset))
    val_size = int(val_ratio * len(syn_dataset))
    test_size = len(syn_dataset) - train_size - val_size

    gen = torch.Generator().manual_seed(42)
    syn_train_dataset, syn_val_dataset, syn_test_dataset = \
            torch_data.random_split(syn_dataset, [train_size, val_size, test_size], generator=gen)
    
    return syn_train_dataset, syn_val_dataset, syn_test_dataset


def get_mnist_dataset_as_specialist(datapath: str = "data/", specialist_class_number: int = 1) -> torch_data.Dataset: 
	transform = torchvision.transforms.Compose([
									torchvision.transforms.Resize(32, antialias=True),
									torchvision.transforms.ToTensor(),
									torchvision.transforms.Normalize((0.5,), (0.5,))])
	
	train = torchvision.datasets.MNIST(root=datapath, train=True, download=True, transform=transform)
	
	targets_idx = train.targets.detach().clone() == specialist_class_number

	return torch_data.dataset.Subset(train, np.where(targets_idx)[0])


if __name__ == '__main__':

    base_dir = './data/4classes'

    custom_dataset = init_four_classes_dataset(base_dir)

    fold_max_samples = []

    n_fold = len(custom_dataset.get_loro())
    for i_fold, (train, val, test) in enumerate(custom_dataset.get_loro()):

        print("### fold ", i_fold, "/", n_fold)

        batch_size = 32
        train_loader = torch_data.DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = torch_data.DataLoader(val, batch_size=batch_size, shuffle=True)
        test_loader = torch_data.DataLoader(test, batch_size=batch_size, shuffle=True)

        lengths = []
        print("\tTrain: ")
        for class_id in train.get_classes():
            print("\t\t", class_id,":", train.get_runs(class_id))

            lengths.append(len(train.get_files(class_id, None)))

        fold_max_samples.append(np.max(lengths))

        print("\tValidação: ")
        for class_id in val.get_classes():
            print("\t\t", class_id,":", val.get_runs(class_id))

        print("\tTest: ")
        for class_id in test.get_classes():
            # print("\t\t", class_id,":", test.get_runs(class_id))
            for run_id in test.get_runs(class_id):
                print("\t\t", class_id,":", run_id,":", sorted(test.get_files(class_id, run_id)))

    print(fold_max_samples)

