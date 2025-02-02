import math
from numba import cuda
import numpy as np
import time

@cuda.jit(device=True)
def get_vector(result_vec, vec, idx):
    result_vec[0] = vec[idx + 0]
    result_vec[1] = vec[idx + 1]
    result_vec[2] = vec[idx + 2]

@cuda.jit(device=True)
def init_vector(result_vec):
    result_vec[0] = 0.0
    result_vec[1] = 0.0
    result_vec[2] = 0.0

@cuda.jit(device=True)
def add_vectors(result_vec, vec1, vec2):
    result_vec[0] = vec1[0] + vec2[0]
    result_vec[1] = vec1[1] + vec2[1]
    result_vec[2] = vec1[2] + vec2[2]

@cuda.jit(device=True)
def subtract_vectors(result_vec, vec1, vec2):
    result_vec[0] = vec1[0] - vec2[0]
    result_vec[1] = vec1[1] - vec2[1]
    result_vec[2] = vec1[2] - vec2[2]

@cuda.jit(device=True)
def dot_vectors(vec1, vec2):
    product = np.zeros(3, dtype=np.float32)
    product[0] = vec1[0] * vec2[0]
    product[1] = vec1[1] * vec2[1]
    product[2] = vec1[2] * vec2[2]
    return product[0] + product[1] + product[2]

@cuda.jit(device=True) 
def cross_vectors(result_vec, vec1, vec2):
    result_vec[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    result_vec[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    result_vec[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]

@cuda.jit(device=True)
def norm(vec):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

@cuda.jit(device=True)
def copy_vector(result_vec, vec, idx):
    result_vec[idx + 0] = vec[0]
    result_vec[idx + 1] = vec[1]
    result_vec[idx + 2] = vec[2]

@cuda.jit(device=True)
def normalize_vector(result_vec, original_vec):
    original_vec_mag = math.sqrt(original_vec[0] * original_vec[0] + original_vec[1] * original_vec[1] + original_vec[2] * original_vec[2])
    if original_vec_mag == 0.0:
        result_vec[0] = 0.0
        result_vec[1] = 0.0
        result_vec[2] = 0.0
    else:
        result_vec[0] = original_vec[0] / original_vec_mag
        result_vec[1] = original_vec[1] / original_vec_mag
        result_vec[2] = original_vec[2] / original_vec_mag

@cuda.jit(device=True)
def scale_vector(result_vec, original_vec, scalar):
    result_vec[0] = original_vec[0] * scalar
    result_vec[1] = original_vec[1] * scalar
    result_vec[2] = original_vec[2] * scalar


@cuda.jit 
def compute_obstacle_heuristic_force_at_anchor(
    anchor, masses, num_masses, mass_radius, detect_radius, force_result):
    # computes force contribution of each mass at anchor
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if idx >= num_masses:
        return

    mass_vector = cuda.local.array(3, dtype=np.float32)
    get_vector(mass_vector, masses, 3*idx)

    # calculate distance and distance vector
    distance_vector = cuda.local.array(3, dtype=np.float32)
    subtract_vectors(distance_vector, mass_vector, anchor)
    distance = 0.0
    distance = norm(distance_vector)


    # compute if radius touches detect radius
    if distance - mass_radius < detect_radius:
        force_vector = cuda.local.array(3, dtype=np.float32)
        distance_unit_vector = cuda.local.array(3, dtype=np.float32)
        
        # heuristic goes here
        normalize_vector(distance_unit_vector, distance_vector)
        force_magnitude = 1.0 / (distance * distance) if distance > 0.001 else 1000.0
        force_magnitude = 1.0*force_magnitude
        scale_vector(force_vector, distance_unit_vector, force_magnitude)
        # heuristic end

        copy_vector(force_result, force_vector, 3*idx)

    else:
        force_vector = cuda.local.array(3, dtype=np.float32)
        init_vector(force_vector)
        copy_vector(force_result, force_vector, 3*idx)


start = time.time()

masses = points.flatten()
num_masses = points.shape[0]
mass_radius = 1.0
detect_radius = 50.0
d_masses = cuda.to_device(masses)
threads_per_block = 256
blocks_per_grid = (num_masses + threads_per_block - 1) // threads_per_block

anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
d_anchor = cuda.to_device(anchor)
force_result = np.zeros(len(masses), dtype=np.float32)
d_force_result = cuda.to_device(force_result)

end = time.time()
print(f"Time taken to load into memory: {end - start} seconds")

start = time.time()
compute_obstacle_heuristic_force_at_anchor[blocks_per_grid, threads_per_block](d_anchor, d_masses, num_masses, mass_radius, detect_radius, d_force_result)
end = time.time()
print(f"Time taken to compute force: {end - start} seconds")

start = time.time()
force_result = d_force_result.copy_to_host()
end = time.time()
print(f"Time taken to copy to host: {end - start} seconds")


kernel_time = 0.0
forces = dict()

for i in range(50):
    for j in range(50):
        for k in range(50):
            anchor = np.array([float(i), float(j), float(k)], dtype=np.float32)
            d_anchor = cuda.to_device(anchor)
            force_result = np.zeros(len(masses), dtype=np.float32)
            d_force_result = cuda.to_device(force_result)

            end = time.time()
            # print(f"Time taken to load into memory: {end - start} seconds")

            start = time.time()
            compute_obstacle_heuristic_force_at_anchor[blocks_per_grid, threads_per_block](d_anchor, d_masses, num_masses, mass_radius, detect_radius, d_force_result)
            end = time.time()
            # print(f"Time taken to compute force: {end - start} seconds")
            kernel_time += end - start

            start = time.time()
            force_result = d_force_result.copy_to_host()
            end = time.time()
            # print(f"Time taken to copy to host: {end - start} seconds")

            # forces.append(force_result.reshape(-1, 3).sum(axis=0))
            forces[(i, j, k)] = force_result.reshape(-1, 3).sum(axis=0)

print(f"Kernel time: {kernel_time} seconds")