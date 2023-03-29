import os
import torch
from torch import nn
import torchvision.utils as vutils

# Set parameters
nz = 100 # size of the noise vector
ngf = 64 # number of generator filters
nc = 3 # number of channels in generated images
output_dir = 'generated_images' # folder to save generated images
num_images = 10 # number of images to generate

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Check if the output directory exists, create it if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load pre-trained weights
device = torch.device('cpu') # load model on CPU
state_dict = torch.load('anime_generator.pt', map_location=device)
netG = Generator(0).to(device)
netG.load_state_dict(state_dict)
netG.eval()

# Generate images
for i in range(num_images):
    # Generate random noise
    noise = torch.randn(1, nz, 1, 1, device=device)

    # Generate image from noise
    with torch.no_grad():
        fake = netG(noise).detach().cpu()

    # Save image to disk
    filename = f"{output_dir}/fake_{i+1}.png"
    try:
        vutils.save_image(fake, filename, normalize=True)
        print(f"Saved image {filename}")
    except FileNotFoundError:
        print(f"Error: could not save file {filename} because the directory does not exist.")

print(f"{num_images} images saved in {output_dir} folder.")
