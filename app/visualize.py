import os, tqdm
import typing
import numpy as np

import app.config as config
import labsonar_ml.utils.utils as ml_utils
import labsonar_ml.data_loader as ml_data
import labsonar_ml.utils.visualization as ml_vis


def append_label_and_data(data: typing.Optional[np.array],
                          labels: typing.List[str],
                          files: typing.List[str],
                          label_id: str,
                        ):

    if files:
        images = ml_data.read_images(files, ml_data.get_defaul_transform(), linearize=True)

        if data is None:
            data = images
        else:
            data = np.concatenate((data, images), axis=0)

        labels.extend([label_id] * images.shape[0])

    return data, labels


def run(reset: bool = False,
        backup: bool = False,
        train: bool = True,
        evaluate: bool = True,
        one_fold_only: bool = False,
        one_class_only: bool = False,
        skip_folds: typing.List[int] = [],
        skip_class: typing.List[str] = [],
        ):

    training_dicts = [{
            'dir': config.Training.GAN,
            'plot_id': str(config.Training.GAN),
            'file_id': str(config.Training.GAN),
        }]

    for sd_reg_factor in [1, 0.8, 0.6, 0.4, 0.2]:
        training_dicts.append({
                'dir': [config.Training.SPEC_GAN, f"{sd_reg_factor}"],
                'plot_id': str(config.Training.SPEC_GAN),
                'file_id': str(config.Training.SPEC_GAN) + "_" + f"{sd_reg_factor}",
            })


    plot_dir = config.get_result_dir([config.Training.PLOTS, "visualization"])

    if evaluate and reset:
        ml_utils.prepare_train_dir(config.get_result_dir(plot_dir), backup=backup)
    os.makedirs(plot_dir, exist_ok=True)

    for i_fold, (train_dataset, val_dataset, test_dataset) in tqdm.tqdm(enumerate(config.get_dataset_loro())):

        if i_fold in skip_folds:
            continue

        real_data = None
        real_labels = []

        for class_id in train_dataset.get_classes():

            files = train_dataset.get_files(class_id=class_id, run_id=None)
            real_data, real_labels = append_label_and_data(real_data, real_labels, files, f"{class_id}_real")

        # ml_vis.export_tsne(real_data, np.array(real_labels), filename=os.path.join(plot_dir, f"t-sne_{i_fold}.png"))
        # ml_vis.export_pca(real_data, np.array(real_labels), filename=os.path.join(plot_dir, f"pca_{i_fold}.png"))

        for training_dict in training_dicts:

            data = real_data.copy()
            labels = real_labels.copy()

            for class_id in train_dataset.get_classes():

                class_dir = config.get_result_dir(training_dict['dir'], artifact=config.Artifacts.OUTPUT, i_fold=i_fold)
                class_dir = os.path.join(class_dir, class_id)
                files = ml_utils.get_files(class_dir, 'png')

                data, labels = append_label_and_data(data, labels, files, f"{class_id}_{str(training_dict['plot_id'])}")

            ml_vis.export_tsne(data, np.array(labels), filename=os.path.join(plot_dir, f"t-sne_{i_fold}-{training_dict['file_id']}.png"))
            # ml_vis.export_pca(data, np.array(labels), filename=os.path.join(plot_dir, f"pca_{i_fold}-{training_dict['file_id']}.png"))


        if one_fold_only:
            break


if __name__ == "__main__":
    ml_utils.print_available_device()
    run()