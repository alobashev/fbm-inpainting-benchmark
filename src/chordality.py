import numpy as np

def find_maximal_submatrix(mask):
    N = mask.shape[0]
    max_size = 0
    best_set = set()
    
    for i in range(N):
        current_set = {i}
        for j in range(N):
            if i != j and mask[i, j] == 0 and all(mask[k, j] == 0 for k in current_set):
                current_set.add(j)
        if len(current_set) > max_size:
            max_size = len(current_set)
            best_set = current_set
            
    return best_set

def find_point_with_max_distances(mask, maximal_set):
    N = mask.shape[0]
    max_known = -1
    best_point = None
    
    for i in range(N):
        if i not in maximal_set:
            known_count = sum(mask[i, j] == 0 for j in maximal_set)
            if known_count > max_known:
                max_known = known_count
                best_point = i
                
    return best_point, max_known

def is_there_unique_solution(mask):
    N = mask.shape[0]
    maximal_set = find_maximal_submatrix(mask)

    while len(maximal_set) < N:
        best_point, max_known = find_point_with_max_distances(mask, maximal_set)
        if max_known < 4:
            return False
        else:
            # Add the best_point to the maximal set and update the mask
            maximal_set.add(best_point)
            mask[best_point, :] = 0
            mask[:, best_point] = 0

    return True
