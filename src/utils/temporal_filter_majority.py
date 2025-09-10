import cupy as cp
from collections import deque
import math

class TemporalMajorityVoxelFilter:
    def __init__(self, window_size=3, logger=None):
        self.window_size = int(window_size)
        self.frames = deque(maxlen=self.window_size)
        self.logger = logger

    def _to_gpu_int32(self, voxel_keys):
        try:
            if isinstance(voxel_keys, cp.ndarray):
                arr = voxel_keys
            else:
                arr = cp.asarray(voxel_keys, dtype=cp.int32)
            
            if arr.ndim == 1:
                arr = arr.reshape(-1, 3)
            
            result = arr.astype(cp.int32, copy=False)
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ERROR in _to_gpu_int32: {e}")
            raise

    def _unique_rows(self, arr):
        try:
            arr = cp.ascontiguousarray(arr)
            
            # For small arrays, use a simple approach
            if arr.shape[0] <= 1000:
                # Convert to tuples for uniqueness check
                tuples = [tuple(row) for row in cp.asnumpy(arr)]
                unique_tuples = list(set(tuples))
                result = cp.asarray(unique_tuples, dtype=cp.int32)
            else:
                # For larger arrays, use a hash-based approach
                # Create a hash for each row
                hash_values = cp.zeros(arr.shape[0], dtype=cp.uint64)
                for i in range(arr.shape[1]):
                    hash_values = hash_values * 1000003 + arr[:, i].astype(cp.uint64)
                
                # Find unique hash values
                unique_hashes, unique_indices = cp.unique(hash_values, return_index=True)
                result = arr[unique_indices]
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ERROR in _unique_rows: {e}")
            raise

    def update(self, voxel_keys):
        try:
            # Convert to GPU array
            frame = self._to_gpu_int32(voxel_keys)
            
            if frame.size == 0:
                self.frames.append(frame.reshape(0, 3))
            else:
                frame = self._unique_rows(frame)
                self.frames.append(frame)

            if not self.frames:
                return []

            # Concatenate frames
            concat = cp.concatenate(list(self.frames), axis=0) if len(self.frames) > 1 else self.frames[0]
            
            if concat.size == 0:
                return []

            # Make contiguous and find unique rows with counts
            concat = cp.ascontiguousarray(concat)
            
            # Use hash-based approach to avoid cp.unique axis=0 issues
            hash_values = cp.zeros(concat.shape[0], dtype=cp.uint64)
            for i in range(concat.shape[1]):
                hash_values = hash_values * 1000003 + concat[:, i].astype(cp.uint64)
            
            # Find unique hashes and their counts
            unique_hashes, unique_indices, counts = cp.unique(hash_values, return_index=True, return_counts=True)
            uv = concat[unique_indices]

            # Calculate threshold and filter
            k = len(self.frames)
            thresh = (k + 1) // 2  # ceil(k/2) for binary median
            
            keep_mask = counts >= thresh
            
            if not keep_mask.any():
                return []

            kept = uv[keep_mask]  # uv is already the correct shape, no need to reshape
            
            # return in requested format: list of 3-tuples of ints
            result = [tuple(map(int, t)) for t in cp.asnumpy(kept)]
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ERROR in update(): {e}")
                import traceback
                self.logger.error(f"ERROR traceback: {traceback.format_exc()}")
            raise
