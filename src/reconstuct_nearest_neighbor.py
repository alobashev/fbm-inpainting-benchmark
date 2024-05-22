import numpy as np
from scipy.interpolate import griddata

def reconstuct_nearest_neighbor(data, mask):
    # Convert mask to boolean if it's not
    mask = mask.astype(bool)
    
    # Get coordinates of known and unknown points
    coords_known = np.array(np.nonzero(mask)).T
    coords_unknown = np.array(np.nonzero(~mask)).T
    
    # Get known values
    values_known = data[mask]
    
    # Perform interpolation
    values_unknown = griddata(coords_known, values_known, coords_unknown, method='nearest')
    
    # Fill unknown values in the original data
    data_interp = data.copy()
    data_interp[~mask] = values_unknown
    
    return data_interp