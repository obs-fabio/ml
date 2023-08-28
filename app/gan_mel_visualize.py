import os, tqdm
import numpy as np
import torch

import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.utils.visualization as ml_vis
import labsonar_ml.data_loader as ml_data

types = [ml_gan.Type.GAN, ml_gan.Type.DCGAN]

data_dir = '/tf/ml/data/4classes'
base_input_dir = '/tf/ml/results/'
syntetic_data_dir = 'output'
plot_base_dir = 'plots'
reset=True
backup_old = False

def read_images(files, transform = None):
    images = []
    for file in files:
        image = ml_data.read_image(file, transform)
        image = image.view(-1)
        images.append(image.tolist())
    return np.array(images)

print(ml_utils.print_available_device())

custom_dataset = ml_data.init_four_classes_dataset(data_dir)

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

    for class_id in custom_dataset.get_classes():

        class_output_dir = os.path.join(syntetic_dir,"{:s}".format(class_id))

        files = custom_dataset.get_files(class_id=class_id, run_id=None)
        if files:
            images = read_images(files, transform=custom_dataset.transform)

            if data is None:
                data = images
            else:
                data = np.concatenate((data, images), axis=0)

            label = "{:s}_real".format(class_id)
            labels.extend([label] * images.shape[0])

        files = ml_utils.get_files(class_output_dir, 'png')
        if files:
            images = read_images(files, transform=custom_dataset.transform)

            data = np.concatenate((data, images), axis=0)

            label = "{:s}_{:s}".format(class_id, type.name.lower())
            labels.extend([label] * images.shape[0])

    if data is None:
        continue

    print(data.shape)
    ml_vis.export_tsne(data, np.array(labels), filename=os.path.join(plot_dir, "{:s}_tse.png".format(type.name.lower())))

# ml_vis.export_tsne(all_data, all_labels, filename=os.path.join(base_input_dir, "gans_tse.png"))
