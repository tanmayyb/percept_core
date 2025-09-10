import cupy as cp
import numpy as np
from collections import deque

class TemporalMedianVoxelFilter:
    def __init__(self, window_size=5, threshold=0.5, max_voxels=1000000, logger=None):
        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self.max_voxels = int(max_voxels)
        self.logger = logger
        
        # GPU arrays for vectorized operations
        self.voxel_coords = None  # (N, 3) int32 - all unique voxels seen
        self.voxel_history = None  # (N, window_size) float32 - presence history
        self.voxel_to_idx = {}  # {voxel_key: idx} mapping
        self.current_size = 0
        self.frames = deque(maxlen=self.window_size)
        self.frame_count = 0  # Track total frames processed
        
        # Pre-allocate GPU memory
        self._initialize_arrays()

    def _initialize_arrays(self):
        try:
            self.voxel_coords = cp.zeros((self.max_voxels, 3), dtype=cp.int32)
            self.voxel_history = cp.zeros((self.max_voxels, self.window_size), dtype=cp.float32)
        except Exception as e:
            if self.logger:
                self.logger.error(f"ERROR in _initialize_arrays: {e}")
                import traceback
                self.logger.error(f"ERROR traceback: {traceback.format_exc()}")
            raise

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

    def _hash_voxels(self, voxels):
        try:
            if voxels.size == 0:
                return cp.array([], dtype=cp.uint64)
            
            # Create hash for each voxel using large prime numbers
            hash_values = cp.zeros(voxels.shape[0], dtype=cp.uint64)
            hash_values = hash_values * 1000003 + voxels[:, 0].astype(cp.uint64)
            hash_values = hash_values * 1000003 + voxels[:, 1].astype(cp.uint64)
            hash_values = hash_values * 1000003 + voxels[:, 2].astype(cp.uint64)
            
            return hash_values
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ERROR in _hash_voxels: {e}")
            raise

    def _find_or_add_voxels(self, current_voxels):
        try:
            if current_voxels.size == 0:
                return cp.array([], dtype=cp.int32)
            
            # Hash current voxels
            current_hashes = self._hash_voxels(current_voxels)
            
            # Hash existing voxels
            if self.current_size > 0:
                existing_hashes = self._hash_voxels(self.voxel_coords[:self.current_size])
            else:
                existing_hashes = cp.array([], dtype=cp.uint64)
            
            # Find which voxels are new
            if existing_hashes.size > 0:
                # Use searchsorted for efficient lookup
                existing_hashes_sorted = cp.sort(existing_hashes)
                search_indices = cp.searchsorted(existing_hashes_sorted, current_hashes)
                
                # Check if hashes exist
                valid_mask = (search_indices < existing_hashes_sorted.size) & \
                           (existing_hashes_sorted[cp.minimum(search_indices, existing_hashes_sorted.size - 1)] == current_hashes)
                
                new_mask = ~valid_mask
            else:
                new_mask = cp.ones(current_hashes.size, dtype=cp.bool_)
            
            # Get indices for existing voxels
            existing_indices = cp.full(current_hashes.size, -1, dtype=cp.int32)
            if existing_hashes.size > 0 and valid_mask.any():
                # Map back to original indices
                sort_indices = cp.argsort(existing_hashes)
                valid_search = search_indices[valid_mask]
                valid_indices = sort_indices[cp.minimum(valid_search, sort_indices.size - 1)]
                existing_indices[valid_mask] = valid_indices
            
            # Add new voxels
            new_voxels = current_voxels[new_mask]
            if new_voxels.size > 0:
                new_count = new_voxels.shape[0]
                
                if self.current_size + new_count > self.max_voxels:
                    # Replace oldest voxels (simple circular buffer)
                    start_idx = self.current_size % self.max_voxels
                    end_idx = min(start_idx + new_count, self.max_voxels)
                    
                    if end_idx > start_idx:
                        self.voxel_coords[start_idx:end_idx] = new_voxels[:end_idx-start_idx]
                        self.voxel_history[start_idx:end_idx] = 0.0
                        remaining = new_count - (end_idx - start_idx)
                        if remaining > 0:
                            self.voxel_coords[:remaining] = new_voxels[end_idx-start_idx:]
                            self.voxel_history[:remaining] = 0.0
                    else:
                        self.voxel_coords[start_idx:] = new_voxels[:self.max_voxels-start_idx]
                        self.voxel_history[start_idx:] = 0.0
                        self.voxel_coords[:end_idx] = new_voxels[self.max_voxels-start_idx:]
                        self.voxel_history[:end_idx] = 0.0
                    
                    self.current_size = self.max_voxels
                else:
                    end_idx = self.current_size + new_count
                    self.voxel_coords[self.current_size:end_idx] = new_voxels
                    self.voxel_history[self.current_size:end_idx] = 0.0
                    self.current_size = end_idx
                
                # Update indices for new voxels
                new_indices = cp.arange(self.current_size - new_count, self.current_size, dtype=cp.int32)
                existing_indices[new_mask] = new_indices
            
            return existing_indices
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ERROR in _find_or_add_voxels: {e}")
                import traceback
                self.logger.error(f"ERROR traceback: {traceback.format_exc()}")
            raise

    def _update_history_vectorized(self, voxel_indices):
        try:
            if voxel_indices.size == 0:
                return
            
            # Shift history left (remove oldest frame)
            self.voxel_history[:, :-1] = self.voxel_history[:, 1:]
            
            # Set current frame presence
            self.voxel_history[:, -1] = 0.0  # Reset all to 0
            self.voxel_history[voxel_indices, -1] = 1.0  # Set present voxels to 1
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ERROR in _update_history_vectorized: {e}")
                import traceback
                self.logger.error(f"ERROR traceback: {traceback.format_exc()}")
            raise

    def _calculate_medians_vectorized(self):
        try:
            if self.current_size == 0:
                return []
            
            # Calculate median for each voxel across the window
            # For binary values (0/1), median is equivalent to majority
            medians = cp.median(self.voxel_history[:self.current_size], axis=1)
            
            # For the first few frames, use a more lenient approach
            # This allows voxels to pass through while the filter is building up history
            if self.frame_count < self.window_size:
                # For initial frames, accept any voxel that was present in the current frame
                # Check if the last column (current frame) has any 1s
                current_frame_presence = self.voxel_history[:self.current_size, -1] > 0.0
                active_mask = current_frame_presence
            else:
                # Use median-based filtering for established history
                effective_threshold = self.threshold
                active_mask = medians > effective_threshold
            
            if not active_mask.any():
                return []
            
            # Get active voxel coordinates
            active_indices = cp.where(active_mask)[0]
            active_coords = self.voxel_coords[active_indices]
            
            # Convert to list of tuples
            result = [tuple(map(int, row)) for row in cp.asnumpy(active_coords)]
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ERROR in _calculate_medians_vectorized: {e}")
                import traceback
                self.logger.error(f"ERROR traceback: {traceback.format_exc()}")
            raise

    def update(self, voxel_keys):
        try:
            self.frame_count += 1
            
            # Convert to GPU array
            current_voxels = self._to_gpu_int32(voxel_keys)
            
            # Find or add voxels and get their indices
            voxel_indices = self._find_or_add_voxels(current_voxels)
            
            # Update history vectorized
            self._update_history_vectorized(voxel_indices)
            
            # Calculate medians and return active voxels
            active_voxels = self._calculate_medians_vectorized()
            
            return active_voxels
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ERROR in update(): {e}")
                import traceback
                self.logger.error(f"ERROR traceback: {traceback.format_exc()}")
            raise

    def reset(self):
        try:
            self.current_size = 0
            self.frames.clear()
            self.voxel_to_idx.clear()
            self.frame_count = 0
            if self.voxel_history is not None:
                self.voxel_history.fill(0.0)
        except Exception as e:
            if self.logger:
                self.logger.error(f"ERROR in reset(): {e}")
            raise
