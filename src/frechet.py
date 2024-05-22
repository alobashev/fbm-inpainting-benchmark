import torch
import einops
from torch.nn.functional import adaptive_avg_pool2d
from .inception import InceptionV3

dims = 64

#TODO change this

print(InceptionV3.BLOCK_INDEX_BY_DIM)
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
Model = InceptionV3([block_idx])




def sqrt_newton_schulz(A, numIters):
    dim = A.size(0)
    normA = A.pow(2).sum().sqrt()
    Y = A.div(normA)
    I = torch.eye(dim, device=A.device)
    Z = I.clone()

    for _ in range(numIters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    sA = Y * normA.sqrt()
    return sA

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, numIters=15):
    diff = mu1 - mu2
    offset = torch.eye(sigma1.size(0), device=sigma1.device) * eps
    covmean = sqrt_newton_schulz((sigma1 + offset) @ (sigma2 + offset), numIters=numIters)
    tr_covmean = torch.diagonal(covmean).sum()
    return (diff @ diff + torch.diagonal(sigma1).sum() + torch.diagonal(sigma2).sum() - 2 * tr_covmean)

def get_freshet_statistics(distance_matrices_batch, device = "cuda:0"):
    model = Model.to(device)
    batch = torch.Tensor(distance_matrices_batch)
    batch = einops.repeat(batch, 'b i j -> b c i j', c=3).to(device)
    with torch.no_grad():
        pred = model(batch)[0]
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    act = pred.squeeze(3).squeeze(2)
    mu = torch.mean(act, dim=0)
    sigma = torch_cov(act)
    return mu, sigma

def get_inception_activations(distance_matrices_batch, device = "cuda:0"):
    model = Model.to(device)
    batch = torch.Tensor(distance_matrices_batch)
    batch = einops.repeat(batch, 'b i j -> b c i j', c=3).to(device)
    with torch.no_grad():
        pred = model(batch)[0]
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    act = pred.squeeze(3).squeeze(2)
    return act

def torch_cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=0)
    x = m - m_exp
    cov = 1 / (x.size(0) - 1) * x.t().mm(x)
    return cov

def compute_FID(distance_matrices_batch_a, distance_matrices_batch_b):
    mu1, sigma1 = get_freshet_statistics(distance_matrices_batch_a)
    mu2, sigma2 = get_freshet_statistics(distance_matrices_batch_b)
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_score
