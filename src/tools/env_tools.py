import numpy as np
from scipy.ndimage import gaussian_filter

def is_margin(position, env_size):
    x_to_edge = min(position[0], env_size - position[0])
    y_to_edge = min(position[1], env_size - position[1])
    return np.array([x_to_edge/(env_size/2), y_to_edge/(env_size/2)], dtype=float)

def make_position(species_position, other_position):
        
    arr = np.zeros((9,9),dtype=float)
    c = 4

    position = other_position-species_position
    position = np.round(position)
    position = position[np.all((position >= -c) & (position <= c), axis=1)]
    if position.ndim == 2:
        position[:,0] += c
        position[:,1] = -position[:,1] + c
        position = position.astype(int)
        arr[position[:,1], position[:,0]] = 1.0
        arr = gaussian_filter(arr, sigma=0.75, truncate=8/3, mode='constant', cval=0.0)
    return arr[2:-2, 2:-2]

