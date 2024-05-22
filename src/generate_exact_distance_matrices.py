import numpy as np
from fbm import FBM

def generate_exact_distance_matrices(num_trajectories, trajectory_length):
    # Generate Brownian motion trajectories in 3D (3 coordinates: x, y, z)
    steps_x = np.random.normal(0, 1, (num_trajectories, trajectory_length - 1))
    steps_y = np.random.normal(0, 1, (num_trajectories, trajectory_length - 1))
    steps_z = np.random.normal(0, 1, (num_trajectories, trajectory_length - 1))

    trajectories_x = np.cumsum(steps_x, axis=1)
    trajectories_y = np.cumsum(steps_y, axis=1)
    trajectories_z = np.cumsum(steps_z, axis=1)

    trajectories_x = np.insert(trajectories_x, 0, 0, axis=1)  # Insert 0 at the beginning of each trajectory
    trajectories_y = np.insert(trajectories_y, 0, 0, axis=1)  # Insert 0 at the beginning of each trajectory
    trajectories_z = np.insert(trajectories_z, 0, 0, axis=1)  # Insert 0 at the beginning of each trajectory

    # Combine x, y and z coordinates to form 3D trajectories
    trajectories = np.stack((trajectories_x, trajectories_y, trajectories_z), axis=-1)

    # Vectorized computation of distance matrices
    # Reshape trajectories for broadcasting
    traj_expanded = trajectories[:, np.newaxis, :, :]
    # Compute pairwise squared differences
    squared_diff = np.sum((traj_expanded - traj_expanded.transpose(0, 2, 1, 3)) ** 2, axis=-1)
    # Take the square root of the sum of squared differences to get Euclidean distances
    distance_matrices = np.sqrt(squared_diff)

    return distance_matrices

def generate_fbm_distance_matrices_old(num_trajectories, trajectory_length, hurst):
    # Generate fractional Brownian motion trajectories in 3D (3 coordinates: x, y, z)
    f = FBM(n=trajectory_length - 1, hurst=hurst, length=1, method='daviesharte')

    trajectories_x = np.array([f.fbm() for _ in range(num_trajectories)])  # Generate fractional Brownian motion
    trajectories_y = np.array([f.fbm() for _ in range(num_trajectories)])  # Generate fractional Brownian motion
    trajectories_z = np.array([f.fbm() for _ in range(num_trajectories)])  # Generate fractional Brownian motion

    # Combine x, y and z coordinates to form 3D trajectories
    trajectories = np.stack((trajectories_x, trajectories_y, trajectories_z), axis=-1)

    # Vectorized computation of distance matrices
    # Reshape trajectories for broadcasting
    traj_expanded = trajectories[:, np.newaxis, :, :]
    # Compute pairwise squared differences
    squared_diff = np.sum((traj_expanded - traj_expanded.transpose(0, 2, 1, 3)) ** 2, axis=-1)
    # Take the square root of the sum of squared differences to get Euclidean distances
    distance_matrices = np.sqrt(squared_diff)

    return distance_matrices*7.8910013315428795


def generate_fbm_distance_matrices(num_trajectories, trajectory_length, hurst):
    # Generate fractional Brownian motion trajectories in 3D (3 coordinates: x, y, z)
    f = FBM(n=trajectory_length - 1, hurst=hurst, length=1, method='daviesharte')

    trajectories_x = np.array([f.fbm() for _ in range(num_trajectories)])  # Generate fractional Brownian motion
    trajectories_y = np.array([f.fbm() for _ in range(num_trajectories)])  # Generate fractional Brownian motion
    trajectories_z = np.array([f.fbm() for _ in range(num_trajectories)])  # Generate fractional Brownian motion

    # Scale the trajectories to have the same standard deviation as the increments in regular Brownian motion
    trajectories_x /= np.std(trajectories_x[:, 1:] - trajectories_x[:, :-1])
    trajectories_y /= np.std(trajectories_y[:, 1:] - trajectories_y[:, :-1])
    trajectories_z /= np.std(trajectories_z[:, 1:] - trajectories_z[:, :-1])

    # Combine x, y and z coordinates to form 3D trajectories
    trajectories = np.stack((trajectories_x, trajectories_y, trajectories_z), axis=-1)

    # Vectorized computation of distance matrices
    # Reshape trajectories for broadcasting
    traj_expanded = trajectories[:, np.newaxis, :, :]
    # Compute pairwise squared differences
    squared_diff = np.sum((traj_expanded - traj_expanded.transpose(0, 2, 1, 3)) ** 2, axis=-1)
    # Take the square root of the sum of squared differences to get Euclidean distances
    distance_matrices = np.sqrt(squared_diff)

    return distance_matrices