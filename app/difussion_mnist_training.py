import os, tqdm
import matplotlib.pyplot as plt

import labsonar_ml.synthesizers.diffusion_model.diffusion_trainer as ml_diff
import labsonar_ml.model.base_model as ml_model
import labsonar_ml.utils.utils as ml_utils

types = [ml_diff.Sampling_strategy.DDPI, ml_diff.Sampling_strategy.DDPM]

data_dir = './data/'
base_dir = './test_results/difussion'
output_dir = 'output'
training_dir = 'training'
batch_size = 16
n_epochs=25
n_samples=100
lr = 1e-3
reset=False
backup_old = True
train = True
evaluate = False


output_dir = os.path.join(base_dir, output_dir)
training_dir = os.path.join(base_dir, training_dir)

if train:

    if reset:
        ml_utils.prepare_train_dir(base_dir, backup=backup_old)
    else:
        os.makedirs(base_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)

    for class_id in tqdm.tqdm(range(1), desc="Class"):
        trainer_file = os.path.join(training_dir, 'trainer_{:d}.plk'.format(class_id))
        training_history_file = os.path.join(training_dir, 'training_history_{:d}.png'.format(class_id))

        if os.path.exists(trainer_file) and \
            os.path.exists(training_history_file):
            continue

        train = ml_utils.get_mnist_dataset_as_specialist(datapath = data_dir, specialist_class_number = class_id)
        
        trainer = ml_diff.DiffusionModel(batch_size=batch_size,
                                         n_epochs = n_epochs,
                                         lr = lr)
        
        errors = trainer.fit(data = train)#, export_progress_file=os.path.join(training_dir, "training_history_{:d}.gif".format(class_id)))

        trainer.save(trainer_file)
        epochs = range(1, errors.shape[0] + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, errors[:,0], label='Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Difussion Errors per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(training_history_file)
        plt.close()

if evaluate:

    os.makedirs(output_dir, exist_ok=True)

    for class_id in tqdm.tqdm(range(1), desc="Class"):

        trainer_file = os.path.join(training_dir, 'trainer_{:d}.plk'.format(class_id))
        
        if not os.path.exists(trainer_file):
            continue

        class_output_dir = os.path.join(output_dir,"{:d}".format(class_id))
        os.makedirs(class_output_dir, exist_ok=True)

        trainer = ml_model.Serializable.load(trainer_file)
        images = trainer.generate_images(n_samples=n_samples)

        for index, image in enumerate(images):
            image_file = os.path.join(class_output_dir, '{:d}.png'.format(index))
            image.save(image_file)
