import numpy as np
from numba import njit

@njit
def compute_radial_distances(anchor:np.ndarray, points:np.ndarray):
    """
    Compute distances from an anchor point to a set of 3D points.

    Parameters:
        anchor (np.ndarray): A 1D array representing the anchor point (x, y, z).
        points (np.ndarray): A 2D array of shape (N, 3) representing the 3D points.

    Returns:
        np.ndarray: A 1D array of distances of length N.
    """
    n_points = points.shape[0]
    distances = np.empty(n_points, dtype=np.float64)

    for i in range(n_points):
        dx = points[i, 0] - anchor[0]
        dy = points[i, 1] - anchor[1]
        dz = points[i, 2] - anchor[2]
        distances[i] = np.sqrt(dx * dx + dy * dy + dz * dz)

    return distances


@njit
def compute_radial_distance_vectors(anchor:np.ndarray, points:np.ndarray, radius:float):
    """
    Compute distance vectors from anchor point to points within a search radius.

    Parameters:
        anchor (np.ndarray): 1D array of shape (3,) representing the 3D anchor point.
        points (np.ndarray): 2D array of shape (N, 3) representing the set of 3D points.
        radius (float): Search radius.

    Returns:
        np.ndarray: 2D array of distance vectors for points within the radius.
    """
    radius_squared = radius ** 2
    results = []

    for i in range(points.shape[0]):
        # Compute squared distance
        dx = points[i, 0] - anchor[0]
        dy = points[i, 1] - anchor[1]
        dz = points[i, 2] - anchor[2]
        dist_squared = dx * dx + dy * dy + dz * dz

        # Check if within the radius
        if dist_squared <= radius_squared:
            results.append((dx, dy, dz))

    return np.array(results)
