import cupy as cp
import numpy as np
from typing import List, Tuple
import time


class TemporalEMAVoxelFilter:
    """
    GPU-accelerated Exponential Moving Average filter for voxel data.
    Optimized to eliminate per-voxel Python loops in the hot path by using set
    intersection/difference and batched GPU updates.
    """

    def __init__(self,
                 alpha: float = 0.7,
                 decay_factor: float = 0.85,
                 confidence_threshold: float = 0.3,
                 max_voxels: int = 100000,
                 cleanup_interval: int = 100):
        self.alpha = float(alpha)
        self.decay_factor = float(decay_factor)
        self.confidence_threshold = float(confidence_threshold)
        self.max_voxels = int(max_voxels)
        self.cleanup_interval = int(cleanup_interval)

        # GPU arrays
        self.voxel_coords = None      # (N, 3) int32
        self.confidences = None       # (N,) float32

        # CPU-side bookkeeping
        self.voxel_to_idx = {}        # { (x,y,z): idx }
        self.free_indices = []        # recycled indices (stack)
        self.current_size = 0
        self.frame_count = 0

        self._initialize_arrays()

    def _initialize_arrays(self):
        self.voxel_coords = cp.zeros((self.max_voxels, 3), dtype=cp.int32)
        self.confidences = cp.zeros(self.max_voxels, dtype=cp.float32)

    # -------- Batch helpers (minimize Python loops) --------

    def _allocate_indices(self, count: int) -> np.ndarray:
        """Allocate 'count' free slots and return them as a NumPy int64 array."""
        idxs = []
        # Reuse free indices first
        reuse = min(count, len(self.free_indices))
        if reuse:
            # Pop from end (O(1))
            for _ in range(reuse):
                idxs.append(self.free_indices.pop())
        # Grow if needed
        remaining = count - reuse
        if remaining > 0:
            grow = min(remaining, self.max_voxels - self.current_size)
            if grow > 0:
                start = self.current_size
                self.current_size += grow
                idxs.extend(range(start, start + grow))
            # If still need more, replace least confident ones
            still = remaining - grow
            if still > 0:
                # Replace the 'still' least-confident among active
                active_confs = self.confidences[:self.current_size]
                # argsort on GPU, then fetch the first 'still' indices
                replace_order = cp.asnumpy(cp.argsort(active_confs))[:still]
                # remove their voxel keys from CPU map
                if replace_order.size:
                    # Bulk fetch coords -> CPU
                    coords_to_delete = cp.asnumpy(self.voxel_coords[replace_order])
                    for c in coords_to_delete:
                        t = (int(c[0]), int(c[1]), int(c[2]))
                        self.voxel_to_idx.pop(t, None)
                idxs.extend(replace_order.tolist())
        return np.asarray(idxs, dtype=np.int64)

    def _add_voxels_batch(self, voxels: List[Tuple[int, int, int]]):
        """Add multiple voxels at once; write coords/conf on GPU in a single pass."""
        if not voxels:
            return
        idxs = self._allocate_indices(len(voxels))
        # Write coords on GPU in one go
        coords_np = np.asarray(voxels, dtype=np.int32)
        self.voxel_coords[idxs] = cp.asarray(coords_np)
        # Initialize confidences for new voxels on GPU
        self.confidences[idxs] = np.float32(self.alpha)
        # Update CPU map in bulk
        self.voxel_to_idx.update({tuple(v): int(i) for v, i in zip(coords_np, idxs)})

    def _cleanup_low_confidence_voxels(self):
        """Periodic memory cleanup for very low-confidence voxels (batched)."""
        if self.current_size == 0:
            return
        # Identify victims on GPU
        thresh = np.float32(self.confidence_threshold * 0.1)
        low_mask = self.confidences[:self.current_size] < thresh
        low_idx = cp.where(low_mask)[0]
        if low_idx.size == 0:
            return
        low_idx_h = cp.asnumpy(low_idx)
        # Remove from CPU map in bulk
        coords = cp.asnumpy(self.voxel_coords[low_idx_h])
        for c in coords:
            self.voxel_to_idx.pop((int(c[0]), int(c[1]), int(c[2])), None)
        # Mark slots reusable
        self.free_indices.extend(map(int, low_idx_h))
        # Optional: zero out confidence (not strictly required, but keeps stats sane)
        self.confidences[low_idx] = 0.0

    # ------------------- Public API -------------------

    def update(self, current_voxels: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Update the EMA filter with the current frame's voxel observations.
        Returns voxels whose confidence >= confidence_threshold.
        """
        self.frame_count += 1
        if not current_voxels and self.current_size == 0:
            return []

        # ---- CPU set logic (fast, no per-voxel Python loops) ----
        curr_set = set(current_voxels)
        tracked_set = set(self.voxel_to_idx.keys())

        # Voxels present this frame and already tracked
        present_voxels = curr_set & tracked_set
        # Voxels new this frame
        new_voxels = curr_set - tracked_set

        # ---- Mark present (batched) ----
        present_idx = None
        if present_voxels:
            present_idx = np.fromiter((self.voxel_to_idx[v] for v in present_voxels),
                                      dtype=np.int64, count=len(present_voxels))
            # Vectorized confidence update on GPU for present voxels
            self.confidences[present_idx] = (
                np.float32(self.alpha) +
                (1.0 - np.float32(self.alpha)) * self.confidences[present_idx]
            )

        # ---- Decay all others (batched) ----
        if self.current_size > 0:
            # Build mask once on GPU
            mask = cp.zeros(self.current_size, dtype=cp.bool_)
            if present_idx is not None and present_idx.size > 0:
                mask[present_idx] = True
            # Decay where not present
            self.confidences[:self.current_size] = cp.where(
                mask,
                self.confidences[:self.current_size],
                np.float32(self.decay_factor) * self.confidences[:self.current_size]
            )

        # ---- Add new voxels (batched) ----
        if new_voxels:
            # (One bulk write to GPU + one dict update)
            self._add_voxels_batch(list(new_voxels))

        # ---- Threshold & return filtered coords (batched) ----
        if self.current_size == 0:
            filtered_voxels: List[Tuple[int, int, int]] = []
        else:
            valid_mask = self.confidences[:self.current_size] >= np.float32(self.confidence_threshold)
            valid_idx = cp.where(valid_mask)[0]
            if valid_idx.size:
                coords = cp.asnumpy(self.voxel_coords[valid_idx])
                filtered_voxels = [tuple(map(int, c)) for c in coords]
            else:
                filtered_voxels = []

        # ---- Periodic cleanup ----
        if self.frame_count % self.cleanup_interval == 0:
            self._cleanup_low_confidence_voxels()

        return filtered_voxels

    def get_stats(self) -> dict:
        if self.current_size == 0:
            return {
                'tracked_voxels': 0,
                'mean_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'memory_usage': 0,
            }
        conf = self.confidences[:self.current_size]
        return {
            'tracked_voxels': int(self.current_size),
            'mean_confidence': float(cp.mean(conf)),
            'max_confidence': float(cp.max(conf)),
            'min_confidence': float(cp.min(conf)),
            'memory_usage': int(len(self.voxel_to_idx)),  # number of active keys
        }

    def reset(self):
        self.voxel_to_idx.clear()
        self.free_indices.clear()
        self.current_size = 0
        self.frame_count = 0
        self.confidences.fill(0.0)