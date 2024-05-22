from tqdm import tqdm
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler
from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 256
    eval_batch_size = 128  # how many images to sample during evaluation
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


config = TrainingConfig()

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
MODEL_PATH = "/home/alexander/storage/dev/sk_hic/diffusion_fbm/diffusion_checkpoints/"
model_checkpoints = sorted([_ for _ in os.listdir(MODEL_PATH) if ".pth" in _])
model.load_state_dict(torch.load(MODEL_PATH+model_checkpoints[-1]))

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

def generate_unconditional(batch_size=1, n_generation_steps=100)

    eval_noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    eval_noise_scheduler.set_timesteps(n_generation_steps)
    eval_noise_scheduler.alphas_cumprod = eval_noise_scheduler.alphas_cumprod.to("cuda:0")

    noise = torch.randn(bs, 1, 64, 64).to("cuda:0")

    noisy_images = noise_scheduler.add_noise(noise*0, noise, eval_noise_scheduler.timesteps[-1])
    noisy_images = noisy_images.to("cuda:0")

    # ranks_evolution = []
    # matrix_evolution = []
    for t in eval_noise_scheduler.timesteps:
        #t = t.repeat(bs)
        t = t.to("cuda:0")
        # 1. predict noise model_output
        with torch.no_grad():
            noise_pred = model(noisy_images, t, return_dict=False)[0]

        # 2. compute previous image: x_t -> x_t-1
        noise_pred_class =  eval_noise_scheduler.step(noise_pred, t, noisy_images)
        noisy_images = noise_pred_class.prev_sample
    return noisy_images.cpu().numpy()[:,0,:,:]