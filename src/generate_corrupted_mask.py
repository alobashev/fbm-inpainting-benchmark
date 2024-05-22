import numpy as np

def generate_corrupted_mask(sparsity=0.23, size=64):
    CM = np.random.rand(size, size)
    mask = np.triu(np.ones((size, size)), k=1)
    CM = CM * (1-mask)
    CM = (CM+CM.T)
    diag_indices = np.diag_indices_from(CM)  
    CM[diag_indices] = CM[diag_indices] / 2 
    CM = (CM > sparsity * 1.0)
    return CM