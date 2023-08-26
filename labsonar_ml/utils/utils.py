import os
import datetime
import shutil
import torch

def get_available_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def images_to_vectors(images):
    return images.view(images.size(0), -1)

def vectors_to_images(vectors, image_dim):
    return vectors.view(vectors.size(0), *image_dim)

def make_targets(n_samples: int, target: int, device):
    return torch.autograd.variable.Variable(torch.ones(n_samples, 1) * target).to(device)

def prepare_train_dir(basepath: str, backup=True):

    if backup:
        os.makedirs(basepath, exist_ok=True)
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
