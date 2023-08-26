import os, tqdm
import torch.utils.data as torch_data

import labsonar_ml.data_loader as ml_data
import labsonar_ml.synthesizers.gan.gan_trainer as ml_gan

data_dir = '/tf/ml/data/4classes'
output_dir = '/tf/ml/results/gan'
batch_size = 32
latent_space_dim=100
n_epochs=100

custom_dataset = ml_data.init_four_classes_dataset(data_dir)


for class_id, loro in tqdm.tqdm(custom_dataset.get_specialist_loro()):

    specialist_dir = os.path.join(output_dir, class_id)
    os.makedirs(specialist_dir, exist_ok=True)

    for run_index, (train, test) in tqdm.tqdm(enumerate(loro), leave=False):

        g_model_file = os.path.join(specialist_dir, 'generator_%d.plk'.format(run_index))
        d_model_file = os.path.join(specialist_dir, 'discriminator_%d.plk'.format(run_index))

        if os.path.exists(g_model_file) and os.path.exists(d_model_file):
            continue

        trainer = ml_gan.Gan_trainer(latent_space_dim = latent_space_dim, n_epochs = n_epochs)
        g_model, d_model, g_errors, d_errors = trainer.fit(data = train)

        g_model.save(g_model_file)
        d_model.save(d_model_file)



