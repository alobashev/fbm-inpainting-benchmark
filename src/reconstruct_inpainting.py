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
    image_size = 96  # the generated image resolution
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


# Example 
# from src.reconstruct_inpainting import InpaintingModel
# model = InpaintingModel("/home/alexander/storage/dev/sk_hic/diffusion_fbm/diffusion_checkpoints/model_H12_20k_10100.pth")
# model.reconstruct_ddpm_inpainting(dm_corrupted_list[-1],cm_list[-1])

class InpaintingModel:
    def __init__(self, checkpoint_path=None, device = 'cuda:0', class_labels=None):

        self.checkpoint_path = checkpoint_path
        self.config = TrainingConfig()
        self.name = os.path.basename(checkpoint_path)[:-4]
        self.device = device
        self.model = UNet2DModel(
            sample_size=self.config.image_size,  # the target image resolution
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
            num_class_embeds = class_labels,
        )
        
        self.model = self.model.to(self.device)
        self.num_class_embeds = class_labels
        
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device,weights_only=False))
        else:  
            print("Don't forget to load checkpoints")
            
        print("Total parameters:", sum([_.numel() for _ in self.model.parameters()])/1e6)

        
    def load_checkpoints(self, checkpoint_path):
    # checkpoint_path = "/home/alexander/storage/dev/sk_hic/diffusion_fbm/diffusion_checkpoints/model_H12_20k_10100.pth"
        self.model.load_state_dict(torch.load(checkpoint_path))
        print(f"Checkpoint {checkpoint_path} is loaded")
        
    def ddnm(self, corupted_data, mask, labels=None, device="cuda:0", n_generation_steps=250, eta = 0.1, travel_length=1, travel_repeat=1):
        # DDNM Based on DDIM
        # l is the travel length
        # r determines the repeat times
        #  l = r = 1 -> DDRM in noisless case 
        # In the DDNM paper they use l=10, r=3
        self.model = self.model.to(device)
        
        corupted_data = torch.Tensor(corupted_data)
        if labels is not None:
            assert self.num_class_embeds is not None
            labels = torch.tensor(labels).to(device)
            assert torch.any(labels >= self.num_class_embeds) == False
            # labels = labels.to(device)
            labels = labels.expand(corupted_data.shape[0] )
        else:
            assert self.num_class_embeds is None
        if len(corupted_data.shape) < 3:
            corupted_data = corupted_data.unsqueeze(0)
        corupted_data = corupted_data.unsqueeze(1)
        corupted_data = corupted_data.to(device)/45-0.25 # normalization for distance matrices
        
        mask = torch.Tensor(mask)
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(1)
        mask = mask.to(device)
        
        noisy_image_last = None
        x0_pred_last = None
    
    
        #### A AND REVERSE A #### 
        A = lambda z: z*mask
        Ap = A
        #########################

        batch_size = corupted_data.shape[0] 
        
        # batch_size, 1, 96, 96  corupted_data.shape
    
        for ___ in range(1):
            eval_noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
            eval_noise_scheduler.set_timesteps(n_generation_steps)
            eval_noise_scheduler.alphas_cumprod = eval_noise_scheduler.alphas_cumprod.to(device)
            
            # Create a tensor of shape (1, 1, 1000)
            schd = (1-eval_noise_scheduler.alphas_cumprod.cpu()).unsqueeze(0).unsqueeze(0)
            schd_resampled = F.interpolate(schd, size=n_generation_steps, mode='linear', align_corners=False)[0,0]
    
            noise = torch.randn(corupted_data.shape).to(device)
    
            x_t = eval_noise_scheduler.add_noise(noise*0, noise, eval_noise_scheduler.timesteps[-1])
            x_t = x_t.to(device)
    
            times  = get_schedule_jump(n_generation_steps, travel_length, travel_repeat)
            time_pairs = list(zip(times[:-1], times[1:]))
            
            for i, j in (time_pairs):
                if i > 0:
                    if j < i:
                    # forward sted
                        t = (torch.ones(batch_size)[0] * i).to(device)

                        # in the paper reversed order so we have to invert here
                        i = n_generation_steps - i
                        j = n_generation_steps - j
                        
                        # 1. predict noise model_output
                        with torch.no_grad():
                            if self.num_class_embeds is not None:
                                noise_pred = self.model(x_t, t, return_dict=False, class_labels=labels)[0]
                            else:
                                noise_pred = self.model(x_t, t, return_dict=False)[0]
                        # 2. compute previous image: x_t -> x_t-1
                        x_t = eval_noise_scheduler.step(noise_pred, t.long(), x_t).prev_sample
            
                        if i < len(eval_noise_scheduler.timesteps) - 1:
                            
                            # get diffusion coefficients
                            timesteps = torch.tensor([eval_noise_scheduler.timesteps[i]])
                            timesteps_next = torch.tensor([eval_noise_scheduler.timesteps[j]])
            
                            at = eval_noise_scheduler.alphas_cumprod[timesteps]
                            at_next =  eval_noise_scheduler.alphas_cumprod[timesteps_next]

                    
                            # First get the prediction of denoised
                            x0_t = (x_t - noise_pred * (1 - at).sqrt()) / at.sqrt() # eq.12

                            # Adjust according to the null space
                            x0_t_hat = x0_t - Ap(A(x0_t) - corupted_data) # eq.13

                            # Sample next x_t
                            # DDIM strategy
                            # eq.52 
                            c1 = (1 - at_next).sqrt() * eta
                            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5) 
                            x_t = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * noise_pred
                            
                            x0_pred_last = x0_t.to('cpu') # store for repaint step
                    else:
                        # backward step for repaint strategy 
                        j = n_generation_steps - j
                        timesteps_next = torch.tensor([eval_noise_scheduler.timesteps[j]])
                        at_next =  eval_noise_scheduler.alphas_cumprod[timesteps_next]
                        x0_t = x0_pred_last.to(device)
                        x_t = at_next.sqrt() * x0_t +  torch.randn_like(x0_t) * (1 - at_next).sqrt()
                        
                # noisy_image_last = x_t.cpu()
    
        #  Fix last output masked region with ground truth 
        noisy_image_last = x_t - Ap(A(x_t) - corupted_data) 
        noisy_image_last = noisy_image_last.cpu().numpy()[:,0]
        return (noisy_image_last + 0.25) * 45 # unnormalization for distance 
        
    
    
# for time travel blocks in ddnm
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        