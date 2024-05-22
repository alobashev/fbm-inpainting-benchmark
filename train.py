import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from src.generate_exact_distance_matrices import generate_fbm_distance_matrices

@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 64
    eval_batch_size = 64  # how many images to sample during evaluation
    num_epochs = 150
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "asep_statess-128"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

class DistanceMatrixDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, hurst=0.5):
        self.hurst = hurst

    def __len__(self):
        return 200000

    def __getitem__(self, idx):

        sample = generate_fbm_distance_matrices(num_trajectories=1, trajectory_length=64, hurst=self.hurst)[0]
        sample = torch.FloatTensor(sample)
        sample = sample.unsqueeze(0)/45-0.25

        return sample

class SavedDistanceMatrixDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path="diffusion_fbm/data/H_2_1/distance_matrices_correct_size_200000.npy"):
        self.data_path = data_path
        self.data_array = np.load(data_path)

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):

        sample = self.data_array[idx]
        sample = torch.FloatTensor(sample)
        sample = sample.unsqueeze(0)/45-0.25

        return sample

config = TrainingConfig()
dataset = SavedDistanceMatrixDataset()
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    class_embed_type = None,
)

model = model.to("cuda:0")
model = torch.nn.DataParallel(model, device_ids=[0,1,2])
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(train_dataloader))
    progress_bar.set_description(f"Epoch {epoch}")

    for step, clean_images in enumerate(train_dataloader):
        clean_images =  clean_images.to("cuda:0")
        #label = _[1].to("cuda:0")

        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)


        # Predict the noise residual
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        loss = F.mse_loss(noise_pred, noise)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)

    PATH  = f"diffusion_checkpoints/model_H12_{10000+epoch}.pth"
    torch.save(model.module.state_dict(), PATH)