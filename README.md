# PyTorch GAN Anime Character Generator

This is a PyTorch implementation of a generative adversarial network (GAN) that generates anime character images. The GAN is trained on a dataset of anime character images and is able to generate new images that are similar to the ones in the dataset.

## Requirements

To run the GAN, you will need:

- Python 3.6 or higher
- PyTorch 1.7 or higher
- torchvision
- matplotlib

You can install the required packages using `pip`:

``` pip install torch torchvision matplotlib ```


## Usage

To generate anime character images using the pre-trained GAN, follow these steps:

1. Clone the repository: https://github.com/G-r-ay/Anime-GAN.git


2. Run the `generate_images.py` script: python generate_images.py

## Training

To train your own PyTorch GAN to generate anime character images, follow these steps:

1. Organize your dataset into a folder containing image files (JPEG or PNG). Each image should be of the same size (e.g., 64x64 pixels).

2. Clone or download the [GitHub repository](https://github.com/G-r-ay/Anime-GAN.git) that contains the GAN code.

3. Edit the `config.py` file in the repository to set the parameters for training, such as the path to your dataset folder, the number of epochs, and the batch size.

4. Run the `train.py` script using the command `python train.py`.

5. After training is complete, the script will save the generator and discriminator models in the `models` directory.

## Generating Images

To generate anime character images using the trained GAN, follow these steps:

1. Create a `anime_generator.pt` file containing the state dict of the generator model.

2. Copy the `generate_images.py` script from the repository to your local machine.

3. Edit the script to set the `netG.pth` file path, the output directory, and the number of images to generate.

4. Run the `generate_images.py` script using the command `python generate_images.py`.

5. The script will generate the specified number of anime character images and save them in the output directory.
