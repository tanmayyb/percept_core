import cupy as cp
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict
import time


class EMAVoxelFilter:
    """
    GPU-accelerated Exponential Moving Average filter for voxel data.
    Handles temporal filtering of voxel coordinates to remove flickering artifacts.
    """
    
    def __init__(self, 
                 alpha: float = 0.7,
                 decay_factor: float = 0.85,
                 confidence_threshold: float = 0.3,
                 max_voxels: int = 100000,
                 cleanup_interval: int = 100):
        """
        Initialize the EMA voxel filter.
        
        Args:
            alpha: Learning rate for new observations (0-1, higher = more responsive)
            decay_factor: Decay rate for absent voxels (0-1, higher = longer memory)
            confidence_threshold: Minimum confidence to include voxel in output
            max_voxels: Maximum number of voxels to track (for memory management)
            cleanup_interval: Frames between confidence cleanup operations
        """
        self.alpha = alpha
        self.decay_factor = decay_factor
        self.confidence_threshold = confidence_threshold
        self.max_voxels = max_voxels
        self.cleanup_interval = cleanup_interval
        
        # GPU arrays for tracking voxel states
        self.voxel_coords = None  # (N, 3) array of voxel coordinates
        self.confidences = None  # (N,) array of confidence values
        self.voxel_to_idx = {}   # CPU dict mapping voxel tuple to GPU array index
        self.free_indices = []   # Available indices for new voxels
        self.current_size = 0    # Current number of tracked voxels
        self.frame_count = 0
        
        # Initialize GPU arrays
        self._initialize_arrays()
    
    def _initialize_arrays(self):
        """Initialize GPU arrays for voxel tracking."""
        self.voxel_coords = cp.zeros((self.max_voxels, 3), dtype=cp.int32)
        self.confidences = cp.zeros(self.max_voxels, dtype=cp.float32)
    
    def _add_voxel(self, voxel: Tuple[int, int, int]) -> int:
        """Add a new voxel to tracking arrays."""
        if self.free_indices:
            idx = self.free_indices.pop()
        elif self.current_size < self.max_voxels:
            idx = self.current_size
            self.current_size += 1
        else:
            # Find least confident voxel to replace
            min_idx = int(cp.argmin(self.confidences[:self.current_size]))
            old_voxel = tuple(self.voxel_coords[min_idx].get())
            del self.voxel_to_idx[old_voxel]
            idx = min_idx
        
        self.voxel_coords[idx] = cp.array(voxel, dtype=cp.int32)
        self.confidences[idx] = 0.0
        self.voxel_to_idx[voxel] = idx
        return idx
    
    def _cleanup_low_confidence_voxels(self):
        """Remove voxels with very low confidence to free up memory."""
        if self.current_size == 0:
            return
            
        # Find voxels with confidence below cleanup threshold
        low_conf_mask = self.confidences[:self.current_size] < (self.confidence_threshold * 0.1)
        low_conf_indices = cp.where(low_conf_mask)[0]
        
        if len(low_conf_indices) == 0:
            return
        
        # Remove from CPU mapping and add to free indices
        for idx in low_conf_indices.get():
            voxel = tuple(self.voxel_coords[idx].get())
            if voxel in self.voxel_to_idx:
                del self.voxel_to_idx[voxel]
                self.free_indices.append(idx)
                self.confidences[idx] = 0.0
    
    def update(self, current_voxels: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Update the filter with new voxel observations and return filtered voxels.
        
        Args:
            current_voxels: List of voxel coordinates as (x, y, z) tuples
            
        Returns:
            List of filtered voxel coordinates
        """
        self.frame_count += 1
        
        # Convert current voxels to set for O(1) lookup
        current_voxel_set = set(current_voxels)
        
        # Create masks for batch operations
        present_mask = cp.zeros(self.current_size, dtype=bool)
        
        # Check which tracked voxels are present in current frame
        for voxel, idx in self.voxel_to_idx.items():
            if voxel in current_voxel_set:
                present_mask[idx] = True
        
        # Batch update confidences on GPU
        if self.current_size > 0:
            # Update present voxels: confidence = alpha * 1.0 + (1-alpha) * confidence
            self.confidences[:self.current_size] = cp.where(
                present_mask,
                self.alpha + (1 - self.alpha) * self.confidences[:self.current_size],
                self.decay_factor * self.confidences[:self.current_size]
            )
        
        # Add new voxels that aren't being tracked
        for voxel in current_voxels:
            if voxel not in self.voxel_to_idx:
                idx = self._add_voxel(voxel)
                self.confidences[idx] = self.alpha
        
        # Get filtered voxels (those above confidence threshold)
        if self.current_size > 0:
            valid_mask = self.confidences[:self.current_size] >= self.confidence_threshold
            valid_indices = cp.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                filtered_coords = self.voxel_coords[valid_indices].get()
                filtered_voxels = [tuple(coord) for coord in filtered_coords]
            else:
                filtered_voxels = []
        else:
            filtered_voxels = []
        
        # Periodic cleanup
        if self.frame_count % self.cleanup_interval == 0:
            self._cleanup_low_confidence_voxels()
        
        return filtered_voxels
    
    def get_stats(self) -> dict:
        """Get statistics about the filter state."""
        if self.current_size == 0:
            return {
                'tracked_voxels': 0,
                'mean_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'memory_usage': 0
            }
        
        confidences = self.confidences[:self.current_size]
        return {
            'tracked_voxels': self.current_size,
            'mean_confidence': float(cp.mean(confidences)),
            'max_confidence': float(cp.max(confidences)),
            'min_confidence': float(cp.min(confidences)),
            'memory_usage': len(self.voxel_to_idx)
        }
    
    def reset(self):
        """Reset the filter state."""
        self.voxel_to_idx.clear()
        self.free_indices.clear()
        self.current_size = 0
        self.frame_count = 0
        self.confidences.fill(0.0)


# Example usage and benchmarking
def benchmark_filter():
    """Benchmark the EMA voxel filter with synthetic data."""
    
    # Initialize filter
    filter_obj = CuPyEMAVoxelFilter(
        alpha=0.7,
        decay_factor=0.85,
        confidence_threshold=0.3,
        max_voxels=50000
    )
    
    # Generate synthetic noisy voxel data
    np.random.seed(42)
    base_voxels = [(i, j, k) for i in range(-30, 30, 2) 
                   for j in range(-30, 30, 2) 
                   for k in range(10, 20)]
    
    print(f"Base voxels: {len(base_voxels)}")
    
    # Simulate frames with noise
    frame_times = []
    
    for frame in range(100):
        # Add noise: remove some base voxels, add random ones
        current_voxels = base_voxels.copy()
        
        # Remove 10% of base voxels (simulating noise)
        remove_count = int(len(current_voxels) * 0.1)
        remove_indices = np.random.choice(len(current_voxels), remove_count, replace=False)
        current_voxels = [v for i, v in enumerate(current_voxels) if i not in remove_indices]
        
        # Add 5% random noise voxels
        noise_count = int(len(base_voxels) * 0.05)
        for _ in range(noise_count):
            noise_voxel = (
                np.random.randint(-50, 50),
                np.random.randint(-50, 50),
                np.random.randint(5, 25)
            )
            current_voxels.append(noise_voxel)
        
        # Time the filtering operation
        start_time = time.perf_counter()
        filtered_voxels = filter_obj.update(current_voxels)
        end_time = time.perf_counter()
        
        frame_times.append(end_time - start_time)
        
        if frame % 20 == 0:
            stats = filter_obj.get_stats()
            print(f"Frame {frame}: Input={len(current_voxels)}, "
                  f"Output={len(filtered_voxels)}, "
                  f"Tracked={stats['tracked_voxels']}, "
                  f"Mean Conf={stats['mean_confidence']:.3f}, "
                  f"Time={frame_times[-1]*1000:.2f}ms")
    
    print(f"\nPerformance Summary:")
    print(f"Mean frame time: {np.mean(frame_times)*1000:.2f}ms")
    print(f"Max frame time: {np.max(frame_times)*1000:.2f}ms")
    print(f"Min frame time: {np.min(frame_times)*1000:.2f}ms")
    print(f"FPS capability: {1.0/np.mean(frame_times):.1f}")


if __name__ == "__main__":
    benchmark_filter()