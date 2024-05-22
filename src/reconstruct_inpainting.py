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


# Example 
# from src.reconstruct_inpainting import InpaintingModel
# model = InpaintingModel("/home/alexander/storage/dev/sk_hic/diffusion_fbm/diffusion_checkpoints/model_H12_20k_10100.pth")
# model.reconstruct_ddpm_inpainting(dm_corrupted_list[-1],cm_list[-1])

class InpaintingModel:
    def __init__(self, checkpoint_path=None, device = 'cuda:0'):

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
        )
        
        self.model = self.model.to(self.device)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:  
            print("Don't forget to load checkpoints")
            
        print("Total parameters:", sum([_.numel() for _ in self.model.parameters()])/1e6)

        
    def load_checkpoints(self, checkpoint_path):
    # checkpoint_path = "/home/alexander/storage/dev/sk_hic/diffusion_fbm/diffusion_checkpoints/model_H12_20k_10100.pth"
        self.model.load_state_dict(torch.load(checkpoint_path))
        print(f"Checkpoint {checkpoint_path} is loaded")
        
    def ddnm(self, distance_matrices_corupt, CM, device="cuda:0", n_generation_steps=250, eta = 0.1, travel_length=1, travel_repeat=1):
        # l is the travel length
        # r determines the repeat times
        #  l = r = 1 -> DDRM in noisless case 
        # In the DDNM paper they use l=10, r=3
        self.model = self.model.to(device)
        use_reschedule = False
        GT = torch.Tensor(distance_matrices_corupt)
        if len(GT.shape) < 3:
            GT = GT.unsqueeze(0)
        GT = GT.unsqueeze(1)
        GT = GT.to(device)/45-0.25
        
        init_mask = torch.Tensor(CM)
        if len(init_mask.shape) < 3:
            init_mask = init_mask.unsqueeze(0)
        init_mask = init_mask.unsqueeze(1)
        init_mask = init_mask.to(device)
        
        noisy_images_history = []
        x0_preds = []
    
    
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
            schd_resampled = F.interpolate(schd, size=n_generation_steps, mode='linear', align_corners=False)[0,0]
    
            noise = torch.randn(bs, 1, 64, 64).to(device)
    
            noisy_images = eval_noise_scheduler.add_noise(noise*0, noise, eval_noise_scheduler.timesteps[-1])
            noisy_images = noisy_images.to(device)
    
            times  = get_schedule_jump(n_generation_steps, travel_length, travel_repeat)
            time_pairs = list(zip(times[:-1], times[1:]))
            
            for i, j in (time_pairs):
                # if j<0: j=-1 # FOR WHAT?
                if i > 0:
                    if j < i:
                        t = (torch.ones(bs)[0] * i).to(device)
                        i = n_generation_steps - i
                        j = n_generation_steps - j
                        # 1. predict noise model_output
                        with torch.no_grad():
                            noise_pred = self.model(noisy_images, t, return_dict=False)[0]
            
                        # 2. compute previous image: x_t -> x_t-1
                        noise_pred_class =  eval_noise_scheduler.step(noise_pred, t.long(), noisy_images)
                        noisy_images = noise_pred_class.prev_sample
            
                        if i < len(eval_noise_scheduler.timesteps) - 1:
                            # noised version of the masked image y
                            # y_{t-1} = A(sqrt(a_{t-1} y + sqrt(1 - a_{t-1} Normal) 
                            # noisy_GT_image = eval_noise_scheduler.add_noise(GT, noise, torch.tensor([eval_noise_scheduler.timesteps[i+1]]))
                            timesteps = torch.tensor([eval_noise_scheduler.timesteps[i]])
                            timesteps_next = torch.tensor([eval_noise_scheduler.timesteps[j]])
                            
                            at = eval_noise_scheduler.alphas_cumprod[timesteps]
            
                            at_next =  eval_noise_scheduler.alphas_cumprod[timesteps_next]
                            # TODO carefully look here
                            # noisy_GT_image = A(at.sqrt() * GT + (1-at).sqrt()*noise)
                            
                            x0_t = (noisy_images - noise_pred * (1 - at).sqrt()) / at.sqrt()
            
                            x0_t_hat = x0_t - Ap(A(x0_t) - GT)
            
                            c1 = (1 - at_next).sqrt() * eta
                            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                            
                            noisy_images = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * noise_pred
                            x0_preds.append(x0_t.to('cpu'))
                    else:
                        j = n_generation_steps - j
                        timesteps_next = torch.tensor([eval_noise_scheduler.timesteps[j]])
                        at_next =  eval_noise_scheduler.alphas_cumprod[timesteps_next]
                        x0_t = x0_preds[-1].to(device)
                        noisy_images = at_next.sqrt() * x0_t +  torch.randn_like(x0_t) * (1 - at_next).sqrt()
                noisy_images_history.append(noisy_images.cpu().numpy()[:,0])
                # noisy_GT_image_history.append(noisy_GT_image.cpu().numpy()[0,0])
    
    
        return (noisy_images_history[-1]+0.25)*45
        
    def ddpm(self, distance_matrices_corupt, CM, device="cuda:0", n_generation_steps=250):
        self.model = self.model.to(device)
        use_reschedule = False
        GT = torch.Tensor(distance_matrices_corupt)
        if len(GT.shape) < 3:
            GT = GT.unsqueeze(0)
        GT = GT.unsqueeze(1)
        GT = GT.to(device)/45-0.25
        
        init_mask = torch.Tensor(CM)
        if len(init_mask.shape) < 3:
            init_mask = init_mask.unsqueeze(0)
        init_mask = init_mask.unsqueeze(1)
        init_mask = init_mask.to(device)

        
        
        noisy_images_history = []
        noisy_GT_image_history = []
    
    
        bs = GT.shape[0]
        
        for ___ in range(1):
            eval_noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
            eval_noise_scheduler.set_timesteps(n_generation_steps)
            eval_noise_scheduler.alphas_cumprod = eval_noise_scheduler.alphas_cumprod.to(device)
            
                # Create a tensor of shape (1, 1, 1000)
            schd = (1-eval_noise_scheduler.alphas_cumprod.cpu()).unsqueeze(0).unsqueeze(0)
            schd_resampled = F.interpolate(schd, size=n_generation_steps, mode='linear', align_corners=False)[0,0]
    
            noise = torch.randn(bs, 1, 64, 64).to(device)
    
            noisy_images = eval_noise_scheduler.add_noise(noise*0, noise, eval_noise_scheduler.timesteps[-1])
            noisy_images = noisy_images.to(device)
    
            ranks_evolution = []
            matrix_evolution = []
            for i, t in enumerate(eval_noise_scheduler.timesteps):
    
    
                t = t.to(device)
                # 1. predict noise model_output
                with torch.no_grad():
                    noise_pred = self.model(noisy_images, t, return_dict=False)[0]
    
    
    
                # 2. compute previous image: x_t -> x_t-1
                noise_pred_class =  eval_noise_scheduler.step(noise_pred, t, noisy_images)
                noisy_images = noise_pred_class.prev_sample
    
                if i < len(eval_noise_scheduler.timesteps) - 1:
                    noisy_GT_image = eval_noise_scheduler.add_noise(GT, noise, torch.tensor([eval_noise_scheduler.timesteps[i+1]]))
                    #noisy_images = (1 - init_mask) * noisy_images + init_mask * noisy_GT_image
                    if use_reschedule:
                        noisy_images = noisy_images  + init_mask * (noisy_GT_image -  noisy_images) * schd_resampled[i]
                    else:
                        noisy_images = noisy_images  + init_mask * (noisy_GT_image -  noisy_images)
                #else:
                #    noisy_images = (1 - init_mask) * noisy_images + init_mask * GT
    
                noisy_images_history.append(noisy_images.cpu().numpy()[:,0])
                noisy_GT_image_history.append(noisy_GT_image.cpu().numpy()[:,0])
    
    
        return (noisy_images_history[-1]+0.25)*45

    def repaint(self, distance_matrices_corupt, CM, device="cuda:0", n_generation_steps=250,  travel_repeat = 3):
        
        self.model = self.model.to(device)
        use_reschedule = False
        GT = torch.Tensor(distance_matrices_corupt)
        if len(GT.shape) < 3:
            GT = GT.unsqueeze(0)
        GT = GT.unsqueeze(1)
        GT = GT.to(device)/45-0.25
        
        init_mask = torch.Tensor(CM)
        if len(init_mask.shape) < 3:
            init_mask = init_mask.unsqueeze(0)
        init_mask = init_mask.unsqueeze(1)
        init_mask = init_mask.to(device)
        
        noisy_images_history = []
    
        resampling_times = travel_repeat
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
            schd_resampled = F.interpolate(schd, size=n_generation_steps, mode='linear', align_corners=False)[0,0]
    
            noise = torch.randn(bs, 1, 64, 64).to(device)
    
            noisy_images = eval_noise_scheduler.add_noise(noise*0, noise, eval_noise_scheduler.timesteps[-1])
            noisy_images = noisy_images.to(device)
    
    
            
            for i, t in enumerate(eval_noise_scheduler.timesteps):
    
                t = t.to(device)
                # 1. predict noise model_output
                
                for s in range(resampling_times):
                    
                    with torch.no_grad():
                        noise_pred = self.model(noisy_images, t, return_dict=False)[0]
    
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
        