import math
# import numpy as np
from numba import cuda, float32


# Set the number of threads per block.
TPB = 256

@cuda.jit
def obstacle_heuristic_kernel(anchors, masses, mass_radius, detect_radius, out_forces):
    """
    Each block processes one anchor. Threads in the block loop over all masses,
    adding the force contribution if the mass is close enough to the anchor.
    A shared–memory reduction then produces one net force per anchor.
    
    Parameters:
      anchors: (N_anchors, 3) array containing anchor positions.
      masses:  (N_masses, 3) array containing mass positions.
      mass_radius: scalar defining the radius of each mass.
      detect_radius: detection range (force is computed if (distance - mass_radius) < detect_radius).
      out_forces: (N_anchors, 3) output array where the summed force vector for each anchor is stored.
    """
    anchor_idx = cuda.blockIdx.x  # one block per anchor
    tid = cuda.threadIdx.x
    n_masses = masses.shape[0]

    # Load the anchor's coordinates.
    a0 = anchors[anchor_idx, 0]
    a1 = anchors[anchor_idx, 1]
    a2 = anchors[anchor_idx, 2]

    # Each thread computes a partial force sum.
    partial_x = 0.0
    partial_y = 0.0
    partial_z = 0.0

    # Precompute threshold squared so that we can quickly decide if a mass is too far.
    threshold = detect_radius + mass_radius
    threshold2 = threshold * threshold

    # Loop over masses in a strided manner.
    for m in range(tid, n_masses, cuda.blockDim.x):
        m0 = masses[m, 0]
        m1 = masses[m, 1]
        m2 = masses[m, 2]
        dx = m0 - a0
        dy = m1 - a1
        dz = m2 - a2
        d2 = dx * dx + dy * dy + dz * dz

        if d2 < threshold2:
            d = math.sqrt(d2)
            # Check the detection condition exactly.
            if d - mass_radius < detect_radius:
                # Use a high force if the distance is extremely small.
                if d > 0.001:
                    force = 1.0 / d2
                else:
                    force = 1000.0
                # Compute the unit vector; if d==0, leave it as zero.
                if d > 0.0:
                    ux = dx / d
                    uy = dy / d
                    uz = dz / d
                else:
                    ux = 0.0
                    uy = 0.0
                    uz = 0.0
                partial_x += force * ux
                partial_y += force * uy
                partial_z += force * uz

    # Use shared memory for a block–wise reduction.
    sdata_x = cuda.shared.array(TPB, dtype=float32)
    sdata_y = cuda.shared.array(TPB, dtype=float32)
    sdata_z = cuda.shared.array(TPB, dtype=float32)
    sdata_x[tid] = partial_x
    sdata_y[tid] = partial_y
    sdata_z[tid] = partial_z
    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            sdata_x[tid] += sdata_x[tid + s]
            sdata_y[tid] += sdata_y[tid + s]
            sdata_z[tid] += sdata_z[tid + s]
        cuda.syncthreads()
        s //= 2

    # The first thread writes the final force vector for this anchor.
    if tid == 0:
        out_forces[anchor_idx, 0] = sdata_x[0]
        out_forces[anchor_idx, 1] = sdata_y[0]
        out_forces[anchor_idx, 2] = sdata_z[0]

# # -----------------------------------------------------------------------------
# # Host code.
# # -----------------------------------------------------------------------------

# # Assume the masses and anchors are provided as Python lists of 3D float vectors.
# # For example, here we generate a list of 125000 random masses.
# n_masses = 125000
# masses_list = [[(np.random.rand() - 0.5) * 100.0,
#                 (np.random.rand() - 0.5) * 100.0,
#                 (np.random.rand() - 0.5) * 100.0] for _ in range(n_masses)]

# # And a list of anchors. For instance, a grid with up to 2000 anchors.
# # (Here we use a 20x10x10 grid = 2000 anchors.)
# nx, ny, nz = 20, 10, 10
# anchors_list = [[float(i), float(j), float(k)] 
#                 for i in range(nx) 
#                 for j in range(ny) 
#                 for k in range(nz)]
# n_anchors = len(anchors_list)

# # Convert the Python lists into NumPy arrays of shape (N, 3) and type float32.
# masses_np = np.array(masses_list, dtype=np.float32)
# anchors_np = np.array(anchors_list, dtype=np.float32)

# mass_radius = 1.0
# detect_radius = 50.0

# # Copy the masses and anchors to the device.
# d_masses = cuda.to_device(masses_np)
# d_anchors = cuda.to_device(anchors_np)

# # Allocate device array for the output forces (one 3D vector per anchor).
# d_out_forces = cuda.device_array((n_anchors, 3), dtype=np.float32)

# threads_per_block = TPB
# blocks_per_grid = n_anchors  # one block per anchor

# # Launch the kernel and measure execution time.
# start_gpu = time.time()
# compute_forces_per_anchor[blocks_per_grid, threads_per_block](
#     d_anchors, d_masses, mass_radius, detect_radius, d_out_forces)
# cuda.synchronize()  # wait for the kernel to finish
# end_gpu = time.time()
# print("GPU kernel execution time: {:.4f} seconds".format(end_gpu - start_gpu))

# # Copy the results back to the host.
# start_copy = time.time()
# out_forces = d_out_forces.copy_to_host()
# end_copy = time.time()
# print("Time to copy results to host: {:.4f} seconds".format(end_copy - start_copy))

# # For example, print the force computed at the first anchor.
# print("Force at anchor 0:", out_forces[0])