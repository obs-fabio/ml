import os
import random
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image

input_folder = "./data/4classes"
output_folder = "./data/4classes_aug"


def generate_augmentation(input_dir, output_dir, n_samples):

    ia.seed(1)
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png'))]

    images = []
    for file in sorted(image_files):
        images.append(np.array(Image.open(os.path.join(input_dir, file)).convert('L')))

    seq = iaa.Sequential([
        iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.20))),
        iaa.Sometimes(0.2, iaa.LinearContrast((0.7, 1.3))),
        # iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.1)),
        iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.1)),
        # iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.005*255), per_channel=0.1)),
        iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0.0, 0.01*255), per_channel=0.1)),
        iaa.Sometimes(0.2, iaa.AdditivePoissonNoise(0.01*255)),
        # iaa.Sometimes(0.2, iaa.Multiply((0.8, 1.2), per_channel=0.1)),
        # iaa.Sometimes(0.2, iaa.SaltAndPepper(0.1)),
        # iaa.Sometimes(0.2, iaa.MedianBlur(k=(3, 11))),
        # iaa.Sometimes(0.2, iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))),
        iaa.Sometimes(0.2, iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))),
        iaa.Sometimes(0.2, iaa.LogContrast(gain=(0.8, 1.2))),
        # iaa.Sometimes(0.2, iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.25))),
    ], random_order=True)

    if len(images) >= n_samples:
        image_list = random.sample(images, n_samples)
    else:
        image_list = random.choices(images, k=n_samples)


    images_aug = seq(images=image_list)
    for i, data in enumerate(images_aug):
        image = Image.fromarray(data, mode='L')
        image.save(os.path.join(output_dir, f'__{i}.png'))

    # for i, data in enumerate(images):
    #     image = Image.fromarray(data, mode='L')
    #     image.save(os.path.join(output_dir, f'{i}.png'))


classes = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
inputs = []
for class_id in classes:
    dir = os.path.join(input_folder, class_id)
    runs = [os.path.join(class_id, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    inputs = inputs + runs

outputs = [os.path.join(output_folder, d) for d in inputs]
inputs = [os.path.join(input_folder,d) for d in inputs]

n_samples = 500
for in_dir, out_dir in zip(inputs, outputs):
    os.makedirs(out_dir, exist_ok=True)
    generate_augmentation(in_dir, out_dir, n_samples)
