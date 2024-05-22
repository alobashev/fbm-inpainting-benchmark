from diffusers import UNet2DModel
from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler
import numpy as np
import os

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

MODEL_PATH = "diffusion_fbm/diffusion_checkpoints/"
model_checkpoints = sorted([_ for _ in os.listdir(MODEL_PATH) if ".pth" in _])
# loaded = model_checkpoints[-1]
loaded = 'model_H12_20k_10100.pth'
print(f"checkpoint {MODEL_PATH+loaded} is loaded")
model.load_state_dict(torch.load(MODEL_PATH+loaded))
Model = model
#model.load_state_dict(torch.load("/home/jupyter-alexander/dev/hic/src/model_10149.pth"))
print("total parameters:", sum([_.numel() for _ in model.parameters()])/1e6)

def reconstruct_repaint_inpainting(distance_matrices_corupt, CM, device="cuda:0", n_generation_steps = 250, resampling_times = 10):
    # 
    model = Model.to(device)
    use_reschedule = False
    GT = torch.Tensor(distance_matrices_corupt)
    GT = GT.unsqueeze(0).unsqueeze(0)
    GT = GT.to(device)/45-0.25
    
    init_mask = torch.Tensor(CM)
    init_mask = init_mask.unsqueeze(0).unsqueeze(0)
    init_mask = init_mask.to(device)
    
    noisy_images_history = []


    #### A AND REVERSE A #### 
    A = lambda z: z*init_mask
    Ap = A
    #########################
    
    bs = 1

    for ___ in range(1):
        eval_noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        eval_noise_scheduler.set_timesteps(n_generation_steps)
        eval_noise_scheduler.alphas_cumprod = eval_noise_scheduler.alphas_cumprod.to(device)
        
            # Create a tensor of shape (1, 1, 1000)
        schd = (1-eval_noise_scheduler.alphas_cumprod.cpu()).unsqueeze(0).unsqueeze(0)
        schd_resampled = F.interpolate(schd, size=550, mode='linear', align_corners=False)[0,0]

        noise = torch.randn(bs, 1, 64, 64).to(device)

        noisy_images = eval_noise_scheduler.add_noise(noise*0, noise, eval_noise_scheduler.timesteps[-1])
        noisy_images = noisy_images.to(device)


        
        for i, t in enumerate(eval_noise_scheduler.timesteps):

            t = t.repeat(bs)
            t = t.to(device)
            # 1. predict noise model_output
            
            for s in range(resampling_times):
                
                with torch.no_grad():
                    noise_pred = model(noisy_images, t, return_dict=False)[0]

                # 2. compute previous image: x_t -> x_t-1
                noise_pred_class =  eval_noise_scheduler.step(noise_pred, t, noisy_images)
                noisy_images = noise_pred_class.prev_sample

                                
                if t > 1:
                    e1 = torch.randn_like(noisy_images)
                    e2 = torch.randn_like(noisy_images)
                else:
                    e1 = torch.zeros_like(noisy_images)
                    e2 = torch.zeros_like(noisy_images)
                    
                if i < len(eval_noise_scheduler.timesteps) - 1:
                    # noised version of the masked image y
                    
                    # noisy_GT_image = eval_noise_scheduler.add_noise(GT, noise, torch.tensor([eval_noise_scheduler.timesteps[i+1]]))
                    timesteps = torch.tensor([eval_noise_scheduler.timesteps[i]])
                    timesteps_next = torch.tensor([eval_noise_scheduler.timesteps[i+1]])
                    
                    at = eval_noise_scheduler.alphas_cumprod[timesteps]
                    at_next = eval_noise_scheduler.alphas_cumprod[timesteps_next]
                    
                    bt =  eval_noise_scheduler.betas[timesteps].to(device)
    
                    # y_{t-1} = sqrt(a_{t-1}) y + sqrt(1 - a_{t-1}) Normal
                    noisy_GT_image = at.sqrt() * GT + (1-at) * e1
    
                    # SIGMA T posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    
                    posterior_variance = bt * (1 - at_next) / (1 - at)
                    
                    # x_{t-1} = 1/sqrt(at_noncum) ( x_t - pred_noise * bt/sqrt(1-at)) + sigma_t * e2
                     
                    x0_t = (noisy_images - noise_pred * bt / (1 - at).sqrt() ) / ( 1 - bt ).sqrt() + posterior_variance * e2
    
                    x0_t = x0_t + A( noisy_GT_image - x0_t)
    
                    
                    if t > 1:
                        noisy_images = (1 - bt).sqrt() * x0_t + bt.sqrt() * e2
                    # print('posterior_variance',posterior_variance)    
                    # print('1-at',(1 - at).sqrt())
                    # print('x0_t', x0_t)
            noisy_images_history.append(noisy_images.cpu().numpy()[0,0])


    return (noisy_images_history[-1]+0.25)*45


def reconstruct_repaint_inpainting_batched(distance_matrices_corupt, CM, device="cuda:0", n_generation_steps = 250, resampling_times = 10):
    # 
    model = Model.to(device)
    use_reschedule = False
    GT = torch.Tensor(distance_matrices_corupt)
    GT = GT.unsqueeze(1)
    GT = GT.to(device)/45-0.25
    
    init_mask = torch.Tensor(CM)
    init_mask = init_mask.unsqueeze(1)
    init_mask = init_mask.to(device)
    
    noisy_images_history = []


    #### A AND REVERSE A #### 
    A = lambda z: z*init_mask
    Ap = A
    #########################
    
    bs = GT.shape[0]

    for ___ in range(1):
        eval_noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        eval_noise_scheduler.set_timesteps(n_generation_steps)
        eval_noise_scheduler.alphas_cumprod = eval_noise_scheduler.alphas_cumprod.to(device)
        
            # Create a tensor of shape (1, 1, 1000)
        schd = (1-eval_noise_scheduler.alphas_cumprod.cpu()).unsqueeze(0).unsqueeze(0)
        schd_resampled = F.interpolate(schd, size=550, mode='linear', align_corners=False)[0,0]

        noise = torch.randn(bs, 1, 64, 64).to(device)

        noisy_images = eval_noise_scheduler.add_noise(noise*0, noise, eval_noise_scheduler.timesteps[-1])
        noisy_images = noisy_images.to(device)


        
        for i, t in enumerate(eval_noise_scheduler.timesteps):

            t = t.to(device)
            # 1. predict noise model_output
            
            for s in range(resampling_times):
                
                with torch.no_grad():
                    noise_pred = model(noisy_images, t, return_dict=False)[0]

                # 2. compute previous image: x_t -> x_t-1
                noise_pred_class =  eval_noise_scheduler.step(noise_pred, t, noisy_images)
                noisy_images = noise_pred_class.prev_sample

                                
                if t > 1:
                    e1 = torch.randn_like(noisy_images)
                    e2 = torch.randn_like(noisy_images)
                else:
                    e1 = torch.zeros_like(noisy_images)
                    e2 = torch.zeros_like(noisy_images)
                    
                if i < len(eval_noise_scheduler.timesteps) - 1:
                    # noised version of the masked image y
                    
                    # noisy_GT_image = eval_noise_scheduler.add_noise(GT, noise, torch.tensor([eval_noise_scheduler.timesteps[i+1]]))
                    timesteps = torch.tensor([eval_noise_scheduler.timesteps[i]])
                    timesteps_next = torch.tensor([eval_noise_scheduler.timesteps[i+1]])
                    
                    at = eval_noise_scheduler.alphas_cumprod[timesteps]
                    at_next = eval_noise_scheduler.alphas_cumprod[timesteps_next]
                    
                    bt =  eval_noise_scheduler.betas[timesteps].to(device)
    
                    # y_{t-1} = sqrt(a_{t-1}) y + sqrt(1 - a_{t-1}) Normal
                    noisy_GT_image = at.sqrt() * GT + (1-at) * e1
    
                    # SIGMA T posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    
                    posterior_variance = bt * (1 - at_next) / (1 - at)
                    
                    # x_{t-1} = 1/sqrt(at_noncum) ( x_t - pred_noise * bt/sqrt(1-at)) + sigma_t * e2
                     
                    x0_t = (noisy_images - noise_pred * bt / (1 - at).sqrt() ) / ( 1 - bt ).sqrt() + posterior_variance * e2
    
                    x0_t = x0_t + A( noisy_GT_image - x0_t)
    
                    
                    if t > 1:
                        noisy_images = (1 - bt).sqrt() * x0_t + bt.sqrt() * e2
                    # print('posterior_variance',posterior_variance)    
                    # print('1-at',(1 - at).sqrt())
                    # print('x0_t', x0_t)
            noisy_images_history.append(noisy_images.cpu().numpy()[:,0])


    return (noisy_images_history[-1]+0.25)*45
