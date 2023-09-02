import os, tqdm
import numpy as np

import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.utils.visualization as ml_vis
import app.config as config

trainings_dict = [
    {
        'type': ml_gan.Type.GAN,
        'dir': config.Training.GAN,
    },
    # {
    #     'type': ml_gan.Type.DCGAN,
    #     'dir': config.Training.DCGAN,
    # }
]

reset=True
backup=True
one_fold_only = True

skip_folds = [0, 1]

ml_utils.print_available_device()

for training_dict in tqdm.tqdm(trainings_dict, desc="Tipos"):

    for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro()), desc=f"{training_dict['type'].name.lower()}_Fold", leave=False):

        if i_fold in skip_folds:
            continue

        plot_dir = config.get_result_dir(i_fold, config.Training.PLOTS)

        data = None
        labels = []

        for class_id in train_dataset.get_classes():

            output_dir = config.get_result_dir(i_fold, training_dict['dir'], config.Artifacts.OUTPUT)
            output_dir = os.path.join(output_dir, class_id)

            files = train_dataset.get_files(class_id=class_id, run_id=None)
            if files:
                images = ml_utils.read_images(files, transform=train_dataset.transform)

                if data is None:
                    data = images
                else:
                    data = np.concatenate((data, images), axis=0)

                label = "{:s}_real".format(class_id)
                labels.extend([label] * images.shape[0])

            files = ml_utils.get_files(output_dir, 'png')
            if files:
                images = ml_utils.read_images(files, transform=train_dataset.transform)

                data = np.concatenate((data, images), axis=0)

                label = f"{class_id}_{training_dict['type'].name.lower()}"
                labels.extend([label] * images.shape[0])

        if data is None:
            continue

        ml_vis.export_tsne(data, np.array(labels), filename=os.path.join(plot_dir, f"{training_dict['type'].name.lower()}_{i_fold}_tse.png".format()))
        # ml_vis.export_pca(data, np.array(labels), filename=os.path.join(plot_dir, f"{training_dict['type'].name.lower()}_{i_fold}_pca.png".format()))

        if one_fold_only:
            break
    # ml_vis.export_tsne(all_data, all_labels, filename=os.path.join(base_input_dir, "gans_tse.png"))
