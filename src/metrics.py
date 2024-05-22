import numpy as np

def compute_RMSE(distance_matrix_a, distance_matrix_b):
    RMSE = np.sqrt(np.mean((distance_matrix_a-distance_matrix_b)**2))
    return RMSE

def compute_MAE(distance_matrix_a, distance_matrix_b):
    MAE = np.mean(np.abs(distance_matrix_a-distance_matrix_b))
    return MAE

def compute_rank(M):
    u, s, v = np.linalg.svd(M)
    s = np.abs(np.sort(s))
    return s[-5:].sum()/s.sum()