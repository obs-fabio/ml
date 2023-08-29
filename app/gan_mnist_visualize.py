import os, tqdm
import numpy as np
import torch.utils.data as torch_data
import torchvision

import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.utils.visualization as ml_vis
import labsonar_ml.data_loader as ml_data

types = [ml_gan.Type.GAN, ml_gan.Type.DCGAN]

data_dir = '/tf/ml/data/'
base_input_dir = '/tf/ml/test_results/'
syntetic_data_dir = 'output'
plot_base_dir = 'plots'
reset=True
backup_old = False

n_samples = 100

transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(32, antialias=True),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Grayscale(),
                                    torchvision.transforms.Normalize((0.5,), (0.5,))])
    

def read_images(files, transform = None):
    images = []
    for file in files:
        image = ml_data.read_image(file, transform)
        image = image.view(-1)
        images.append(image.tolist())
    return np.array(images)

ml_utils.print_availprintable_device()

all_data = []
all_labels = []

for type in types:

    type_dir = os.path.join(base_input_dir, type.name.lower())
    syntetic_dir = os.path.join(type_dir, syntetic_data_dir)
    plot_dir = os.path.join(type_dir, plot_base_dir)

    if reset:
        ml_utils.prepare_train_dir(plot_dir, backup=backup_old)
    else:
        os.makedirs(plot_dir, exist_ok=True)

    data = None
    labels = []

    for class_id in tqdm.tqdm(range(10), desc="Class"):

        class_output_dir = os.path.join(syntetic_dir,"{:d}".format(class_id))

        train = ml_utils.get_mnist_dataset_as_specialist(datapath = data_dir, specialist_class_number = class_id)
        data_loader = torch_data.DataLoader(train, batch_size=n_samples, shuffle=True)

        for samples, _ in data_loader:
            images = samples.view(samples.size(0), -1).numpy()
            
            label = "{:d}_real".format(class_id)
            labels.extend([label] * images.shape[0])

            if data is None:
                data = images
            else:
                data = np.concatenate((data, images), axis=0)
            break

        files = ml_utils.get_files(class_output_dir, 'png')
        if files:
            images = read_images(files, transform=transform)
            data = np.concatenate((data, images), axis=0)

            label = "{:d}_{:s}".format(class_id, type.name.lower())
            labels.extend([label] * images.shape[0])

    if data is None:
        continue

    ml_vis.export_tsne(data, np.array(labels), filename=os.path.join(plot_dir, "{:s}_tse.png".format(type.name.lower())))

# ml_vis.export_tsne(all_data, all_labels, filename=os.path.join(base_input_dir, "gans_tse.png"))
