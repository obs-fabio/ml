import os
from typing import Type, List, Dict, Tuple
from PIL import Image
import torchvision
import torch.utils.data as torch_data

def read_image(image_file, transform = None) -> Image:
    image = Image.open(image_file).convert('RGB')
    if transform:
        image = transform(image)
    return image

class Base_dataset (torch_data.Dataset):

    def __init__(self, samples: List[Tuple[str, str, str]], classes: List[str], runs: Dict[str, List[str]], transform):
        super().__init__()
        self._samples = samples
        self._classes = classes
        self._runs = runs
        self.transform = transform

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        image_path, class_id, _ = self._samples[idx]
        image = read_image(image_path, self.transform)
        return image, class_id

    def get_classes(self):
        return self._classes.copy()

    def get_runs(self, class_id: str) -> List[str]:
        return list(self._runs[class_id])

    def get_files(self, class_id: str, run_id: str) -> List[str]:
        files = []
        for image_path, _class_id, _run_id in self._samples:
            if (class_id == _class_id and ((run_id == _run_id) or run_id == None)) or class_id == None:
                files.append(image_path)
        return files


class Dataset_manager (Base_dataset):

    def __init__(self, samples: List[Tuple[str, str, str]], classes: List[str], runs: Dict[str, List[str]], transform):
        super().__init__(samples, classes, runs, transform)

    def get_loro(self, class_id = None) -> List[Tuple[Type[Base_dataset], Type[Base_dataset]]]:

        max_runs = 0
        selected_classes = self._classes if class_id == None else [class_id]
        for class_id in selected_classes:
            max_runs = max(max_runs, len(self._runs[class_id]))

        subsets = []
        for run_index in range(max_runs):
            test = []
            train = []
            train_runs = {}
            test_runs = {}

            for _class_id in selected_classes:

                if _class_id not in selected_classes:
                    continue

                train_runs[_class_id] = [run for index, run in enumerate(self._runs[_class_id])
                                         if run_index % len(self._runs[_class_id]) != index]

                test_runs[_class_id] = [self._runs[_class_id][run_index % len(self._runs[_class_id])]]

            for image_path, _class_id, _run_id in self._samples:

                if _class_id not in selected_classes:
                    continue

                if (run_index % len(self._runs[_class_id])) == self._runs[_class_id].index(_run_id):
                    test.append([image_path, _class_id, _run_id])
                else:
                    train.append([image_path, _class_id, _run_id])

            subsets.append([Base_dataset(train, selected_classes, train_runs, self.transform),
                            Base_dataset(test, selected_classes, test_runs, self.transform)])

        return subsets

    def get_specialist_loro(self) -> List[Tuple[str, List[Tuple[Type[Base_dataset], Type[Base_dataset]]]]]:
        class_datasets = []
        for class_id in self._classes:
            class_datasets.append([class_id, self.get_loro(class_id)])
        return class_datasets


def init_four_classes_dataset(base_dir: str,
        transform: torchvision.transforms.Compose = None,
        extension: str = '.png') -> Type[Dataset_manager]:
    _samples = []
    _classes = []
    _runs = {}

    if transform is None:
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,))
            ])
    else:
        transform = transform

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


if __name__ == '__main__':

    base_dir = '/tf/ml/data/4classes'

    custom_dataset = init_four_classes_dataset(base_dir)

    print("###############")
    for class_id in custom_dataset.get_classes():
        for run_id in custom_dataset.get_runs(class_id):
            for file in custom_dataset.get_files(class_id, run_id):
                print(class_id,"  ", run_id, " : ", file)


    print("\n###############")
    for train, test in custom_dataset.get_loro():

        batch_size = 32
        data_loader = torch_data.DataLoader(train, batch_size=batch_size, shuffle=True)
        data_loader = torch_data.DataLoader(test, batch_size=batch_size, shuffle=True)

        print("\tTrain: ", train.get_classes())
        for class_id in train.get_classes():
            print("\t\t", train.get_runs(class_id))

        print("\tTest: ", test.get_classes())
        for class_id in test.get_classes():
            print("\t\t", test.get_runs(class_id))

            for run_id in test.get_runs(class_id):
                print("\t\t\t", run_id, " -> ", test.__len__() ," -> " , len(test.get_files(class_id, run_id)))

                for files in test.get_files(class_id, run_id):
                    print("\t\t\t\t", files)


    print("\n###############")
    for class_id, loro in custom_dataset.get_specialist_loro():

        print("## Class: ", class_id)
        for train, test in loro:

            batch_size = 32
            data_loader = torch_data.DataLoader(train, batch_size=batch_size, shuffle=True)
            data_loader = torch_data.DataLoader(test, batch_size=batch_size, shuffle=True)

            print("\t\tTrain: ", train.get_classes())
            for class_id in train.get_classes():

                for run_id in train.get_runs(class_id):
                    print("\t\t\t", run_id, " -> ", len(train.get_files(class_id, run_id)))

                    for files in train.get_files(class_id, run_id):
                        print("\t\t\t\t", files)

            print("\t\tTest: ", test.get_classes())
            for class_id in test.get_classes():

                for run_id in test.get_runs(class_id):
                    print("\t\t\t", run_id, " -> ", len(test.get_files(class_id, run_id)))

                    for files in test.get_files(class_id, run_id):
                        print("\t\t\t\t", files)

    print(next(iter(data_loader))[0].size())