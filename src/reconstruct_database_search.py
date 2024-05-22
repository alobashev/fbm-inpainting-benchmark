import numpy as np

def reconstruct_database_search(distance_matrices_correct, CM, database_distance_matrices_correct, top_k=100 ):
    tensor_repeated = np.repeat((distance_matrices_correct*CM)[np.newaxis, :, :], repeats=len(database_distance_matrices_correct), axis=0)
    database_corruption_mask = np.repeat((CM)[np.newaxis, :, :], repeats=len(database_distance_matrices_correct), axis=0)
    error_matrix = np.abs( database_corruption_mask*database_distance_matrices_correct-tensor_repeated)
    error_matrix = np.mean(error_matrix, axis=(1, 2))
    idx = np.argsort(error_matrix)[np.random.randint(0,top_k)]
    M_hat = distance_matrices_correct*CM+(1-CM)*database_distance_matrices_correct[idx]
    return M_hat