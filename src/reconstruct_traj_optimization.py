import torch
from tqdm import tqdm
from torch.optim import Adam

def reconstruct_traj_optimization(distance_matrices, CM):
    
    distance_matrices_CM = torch.Tensor(distance_matrices*CM)
    CM_torch = torch.Tensor(CM).type(torch.bool)

    traj = torch.randn(64,3)
    traj.requires_grad = True
    opt = Adam([traj], lr=0.02)
    n_steps = 1500
    
    loss_history = []
    for i in range(n_steps):
        opt.zero_grad()
        distance_matrices_recon = torch.cdist(traj, traj)
        loss = torch.mean((distance_matrices_recon[CM_torch]-distance_matrices_CM[CM_torch])**2)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
    return distance_matrices_recon.detach().numpy()



def reconstruct_traj_optimization_with_prior(distance_matrices, CM, prior_weight = 15, n_steps = 550,  lr=0.2):
    
    distance_matrices_CM = torch.Tensor(distance_matrices*CM)
    CM_torch = torch.Tensor(CM).type(torch.bool)

    traj = torch.randn(64,3)
    traj.requires_grad = True
    opt = Adam([traj], lr)
    
    loss_history = []
    for i in range(n_steps):
        opt.zero_grad()
        distance_matrices_recon = torch.cdist(traj, traj)
        loss = torch.mean((distance_matrices_recon[CM_torch]-distance_matrices_CM[CM_torch])**2)
        prior_loss = torch.mean(torch.sqrt(torch.sum((traj[1:] - traj[:-1])**2, dim=-1))) / (traj.size(0)-1)
        loss = loss + prior_weight * prior_loss
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
    return distance_matrices_recon.detach().numpy()