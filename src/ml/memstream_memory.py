"""
MemoryModule: Per-Neighborhood FIFO Memory Architecture for MemStream.

Implements 10 independent continuous FIFO buffers (one per neighborhood),
each with 50,000 slots storing AE output tensors at out_dim=34.

Key Design Decisions:
1. Per-neighborhood memory: Enables localized concept drift detection
2. FIFO replacement: Simple, no-complexity eviction policy
3. Graceful degradation: Returns 0.5 if memory is empty
4. L1 kNN scoring: Manhattan distance for anomaly scoring

Memory Update Policy:
- Only normal points update memory (score < beta)
- Anomalies do NOT update memory (preserves memory purity)
- This is the original MemStream paper semantics

Reference: Phase 1C migration plan
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# Neighborhood Definitions (10 neighborhoods)
# =============================================================================

NEIGHBORHOODS: List[str] = [
    'manhattan',       # 0
    'brooklyn',        # 1
    'queens_lower',    # 2
    'queens_upper',    # 3
    'bronx',           # 4
    'staten_island',   # 5
    'ewr',             # 6
    'jfk',             # 7
    'nalp',            # 8
    'unknown',         # 9
]

NUM_NEIGHBORHOODS: int = len(NEIGHBORHOODS)  # 10

NEIGHBORHOOD_TO_ID: Dict[str, int] = {nb: i for i, nb in enumerate(NEIGHBORHOODS)}
ID_TO_NEIGHBORHOOD: Dict[int, str] = {i: nb for i, nb in enumerate(NEIGHBORHOODS)}


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_MEMORY_SLOTS: int = 50_000
DEFAULT_OUT_DIM: int = 34  # v10 benchmark: 34D AE output
DEFAULT_K_NEIGHBORS: int = 10
DEFAULT_GAMMA: float = 0.0  # No decay weighting


# =============================================================================
# NeighborhoodMemory: Single FIFO Buffer
# =============================================================================

class NeighborhoodMemory:
    """
    Single neighborhood's memory buffer (FIFO queue).

    Stores encoded normal patterns from the autoencoder.
    Uses circular buffer for O(1) insertions.
    """

    def __init__(
        self,
        memory_slots: int = DEFAULT_MEMORY_SLOTS,
        out_dim: int = DEFAULT_OUT_DIM,
        device: str = 'cpu',
    ):
        """
        Args:
            memory_slots: Number of FIFO slots (default: 50,000)
            out_dim: AE output dimension (default: 34)
            device: PyTorch device for tensors
        """
        self.memory_slots = memory_slots
        self.out_dim = out_dim
        self.device = device

        # Memory storage: [memory_slots, out_dim]
        self.memory = torch.zeros(
            memory_slots, out_dim,
            dtype=torch.float32,
            device=device
        )

        # Usage tracking: 1.0 = filled, 0.0 = empty
        self.mem_usage = torch.zeros(
            memory_slots,
            dtype=torch.float32,
            device=device
        )

        # Circular pointer (next write position)
        self.mem_ptr = 0

        # Total samples ever added
        self.count = 0

        # Memory fullness tracking
        self._is_full = False
        self._memory_head = 0

    def add(self, ae_output: torch.Tensor) -> None:
        """
        Add AE output to memory (FIFO replacement).

        Args:
            ae_output: Encoded representation [out_dim] or [1, out_dim]
        """
        z = ae_output.detach().clone()

        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Handle batch inserts
        for i in range(min(z.shape[0], self.memory_slots)):
            self.memory[self.mem_ptr] = z[i]
            self.mem_usage[self.mem_ptr] = 1.0
            self.mem_ptr = (self.mem_ptr + 1) % self.memory_slots
            self.count += 1
            self._memory_head += 1

            if self.count >= self.memory_slots:
                self._is_full = True

    @property
    def is_empty(self) -> bool:
        """Check if memory has no samples."""
        return self.count == 0

    @property
    def is_full(self) -> bool:
        """Check if memory is at capacity."""
        return self._is_full

    @property
    def fill_ratio(self) -> float:
        """Memory fill ratio (0.0 to 1.0)."""
        return min(self.count / self.memory_slots, 1.0)

    def get_filled_memory(self) -> torch.Tensor:
        """
        Get memory tensor with only filled slots.

        Returns:
            Tensor of shape [actual_count, out_dim]
        """
        if self.count == 0:
            return torch.zeros(0, self.out_dim, device=self.device)

        if self._is_full:
            return self.memory.clone()

        return self.memory[:self.count].clone()

    def get_all_memory(self) -> torch.Tensor:
        """Get all memory slots (including empty)."""
        return self.memory.clone()

    def reset(self) -> None:
        """Reset memory to empty state."""
        self.memory.zero_()
        self.mem_usage.zero_()
        self.mem_ptr = 0
        self.count = 0
        self._is_full = False
        self._memory_head = 0

    def get_state_dict(self) -> dict:
        """Export state for checkpointing."""
        return {
            'memory': self.memory.cpu(),
            'mem_usage': self.mem_usage.cpu(),
            'mem_ptr': self.mem_ptr,
            'count': self.count,
            'memory_slots': self.memory_slots,
            'out_dim': self.out_dim,
            '_is_full': self._is_full,
            '_memory_head': self._memory_head,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.memory = state['memory'].to(self.device)
        self.mem_usage = state['mem_usage'].to(self.device)
        self.mem_ptr = state.get('mem_ptr', 0)
        self.count = state.get('count', 0)
        self.memory_slots = state.get('memory_slots', DEFAULT_MEMORY_SLOTS)
        self.out_dim = state.get('out_dim', DEFAULT_OUT_DIM)
        self._is_full = state.get('_is_full', False)
        self._memory_head = state.get('_memory_head', 0)


# =============================================================================
# MemoryModule: 10 Independent FIFO Buffers
# =============================================================================

class MemoryModule:
    """
    Memory module with 10 independent FIFO buffers (one per neighborhood).

    Each neighborhood maintains its own memory of encoded normal patterns,
    enabling localized drift detection and context-aware scoring.

    Key Methods:
    - add(neighborhood_id, ae_output): Add sample to neighborhood's memory
    - query(neighborhood_id, ae_output, k, gamma): L1 kNN query

    Scoring:
    - L1 (Manhattan) distance for anomaly scoring
    - k nearest neighbors with optional gamma decay weighting
    - Graceful degradation: returns 0.5 if memory is empty
    """

    def __init__(
        self,
        memory_slots: int = DEFAULT_MEMORY_SLOTS,
        out_dim: int = DEFAULT_OUT_DIM,
        device: str = 'cpu',
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        gamma: float = DEFAULT_GAMMA,
    ):
        """
        Args:
            memory_slots: FIFO slots per neighborhood (default: 50,000)
            out_dim: AE output dimension (default: 34)
            device: PyTorch device
            k_neighbors: k for kNN queries (default: 10)
            gamma: Decay factor for weighted kNN (default: 0.0, no decay)
        """
        self.memory_slots = memory_slots
        self.out_dim = out_dim
        self.device = device
        self.k_neighbors = k_neighbors
        self.gamma = gamma

        # 10 independent neighborhood memories
        self._memories: Dict[int, NeighborhoodMemory] = {
            nb_id: NeighborhoodMemory(
                memory_slots=memory_slots,
                out_dim=out_dim,
                device=device,
            )
            for nb_id in range(NUM_NEIGHBORHOODS)
        }

        # Rolling buffer for recent scores (quick_retrain baseline)
        self._recent_scores: Dict[int, deque] = {
            nb_id: deque(maxlen=5000)
            for nb_id in range(NUM_NEIGHBORHOODS)
        }

        # Statistics
        self._total_adds = 0

    def add(
        self,
        neighborhood_id: int,
        ae_output: torch.Tensor,
    ) -> None:
        """
        Add AE output to neighborhood's memory.

        Args:
            neighborhood_id: Neighborhood index (0-9)
            ae_output: Encoded representation [out_dim]
        """
        nb_id = int(neighborhood_id)
        if 0 <= nb_id < NUM_NEIGHBORHOODS:
            self._memories[nb_id].add(ae_output)
            self._total_adds += 1

    def record_score(
        self,
        neighborhood_id: int,
        score: float,
    ) -> None:
        """
        Record a score for quick_retrain baseline.

        Args:
            neighborhood_id: Neighborhood index (0-9)
            score: Anomaly score to record
        """
        nb_id = int(neighborhood_id)
        if 0 <= nb_id < NUM_NEIGHBORHOODS:
            self._recent_scores[nb_id].append(float(score))

    def query(
        self,
        neighborhood_id: int,
        ae_output: torch.Tensor,
        k: int = None,
        gamma: float = None,
    ) -> float:
        """
        Query memory with L1 kNN distance.

        Computes the sum of L1 distances to the k nearest neighbors
        in the neighborhood's memory.

        Args:
            neighborhood_id: Neighborhood index (0-9)
            ae_output: Encoded representation [out_dim]
            k: Override k neighbors (default: self.k_neighbors)
            gamma: Override decay factor (default: self.gamma)

        Returns:
            L1 kNN distance (higher = more anomalous)
            Returns 0.5 if memory is empty (graceful degradation)
        """
        nb_id = int(neighborhood_id)
        k = k if k is not None else self.k_neighbors
        gamma = gamma if gamma is not None else self.gamma

        if 0 > nb_id >= NUM_NEIGHBORHOODS:
            return 0.5

        memory = self._memories[nb_id]

        # Graceful degradation: empty memory
        if memory.count < 2:
            return 0.5

        # Ensure input is 2D
        z = ae_output.detach().clone()
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Get filled memory
        mem_arr = memory.get_filled_memory()

        if mem_arr.shape[0] < 2:
            return 0.5

        # Compute L1 distances
        mem_np = mem_arr.cpu().numpy()
        z_np = z[0].cpu().numpy()

        dists = np.sum(np.abs(mem_np - z_np), axis=1)

        # Find k nearest
        k_use = min(k, len(dists))
        top_idx = np.argpartition(dists, k_use)[:k_use]
        top_d = np.sort(dists[top_idx])

        # Sum with optional gamma decay
        if gamma > 0:
            score = sum((gamma ** i) * top_d[i] for i in range(k_use))
        else:
            score = float(np.sum(top_d))

        return score

    def query_gpu(
        self,
        neighborhood_id: int,
        ae_output: torch.Tensor,
        k: int = None,
        gamma: float = None,
    ) -> float:
        """
        GPU-accelerated L1 kNN query.

        Args:
            neighborhood_id: Neighborhood index (0-9)
            ae_output: Encoded representation [out_dim]
            k: Override k neighbors
            gamma: Override decay factor

        Returns:
            L1 kNN distance
        """
        nb_id = int(neighborhood_id)
        k = k if k is not None else self.k_neighbors
        gamma = gamma if gamma is not None else self.gamma

        if 0 > nb_id >= NUM_NEIGHBORHOODS:
            return 0.5

        memory = self._memories[nb_id]

        if memory.count < 2:
            return 0.5

        z = ae_output.detach().clone().to(self.device)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        mem_tensor = memory.get_filled_memory()

        if mem_tensor.shape[0] < 2:
            return 0.5

        k_use = min(k, mem_tensor.shape[0])

        # L1 distance: sum of absolute differences
        diff = z - mem_tensor
        dists = diff.abs().sum(dim=1)

        # Top-k
        top_k = dists.topk(k_use, largest=False)
        top_d = top_k.values

        if gamma > 0:
            powers = torch.arange(k_use, device=self.device, dtype=torch.float32)
            weights = gamma ** powers
            score = (top_d.float() * weights).sum().item()
        else:
            score = top_d.sum().item()

        return score

    def get_neighborhood_stats(self, neighborhood_id: int) -> dict:
        """Get statistics for a specific neighborhood."""
        nb_id = int(neighborhood_id)
        if 0 > nb_id >= NUM_NEIGHBORHOODS:
            return {}

        memory = self._memories[nb_id]
        return {
            'neighborhood_id': nb_id,
            'neighborhood_name': ID_TO_NEIGHBORHOOD.get(nb_id, 'unknown'),
            'count': memory.count,
            'capacity': self.memory_slots,
            'fill_ratio': memory.fill_ratio,
            'is_full': memory.is_full,
            'recent_scores_count': len(self._recent_scores[nb_id]),
        }

    def get_stats(self) -> dict:
        """Get statistics for all neighborhoods."""
        per_neighborhood = {
            nb_id: {
                'count': self._memories[nb_id].count,
                'fill_ratio': self._memories[nb_id].fill_ratio,
                'is_full': self._memories[nb_id].is_full,
            }
            for nb_id in range(NUM_NEIGHBORHOODS)
        }

        return {
            'total_adds': self._total_adds,
            'per_neighborhood': per_neighborhood,
        }

    def reset(self, neighborhood_id: int = None) -> None:
        """
        Reset memory.

        Args:
            neighborhood_id: Specific neighborhood to reset, or None for all
        """
        if neighborhood_id is not None:
            nb_id = int(neighborhood_id)
            if 0 <= nb_id < NUM_NEIGHBORHOODS:
                self._memories[nb_id].reset()
                self._recent_scores[nb_id].clear()
        else:
            for nb_id in range(NUM_NEIGHBORHOODS):
                self._memories[nb_id].reset()
                self._recent_scores[nb_id].clear()

    def get_state_dict(self) -> dict:
        """Export state for checkpointing."""
        return {
            'memory_slots': self.memory_slots,
            'out_dim': self.out_dim,
            'k_neighbors': self.k_neighbors,
            'gamma': self.gamma,
            '_total_adds': self._total_adds,
            'neighborhoods': {
                nb_id: self._memories[nb_id].get_state_dict()
                for nb_id in range(NUM_NEIGHBORHOODS)
            },
            'recent_scores': {
                nb_id: list(self._recent_scores[nb_id])
                for nb_id in range(NUM_NEIGHBORHOODS)
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.memory_slots = state.get('memory_slots', DEFAULT_MEMORY_SLOTS)
        self.out_dim = state.get('out_dim', DEFAULT_OUT_DIM)
        self.k_neighbors = state.get('k_neighbors', DEFAULT_K_NEIGHBORS)
        self.gamma = state.get('gamma', DEFAULT_GAMMA)
        self._total_adds = state.get('_total_adds', 0)

        if 'neighborhoods' in state:
            for nb_id, nb_state in state['neighborhoods'].items():
                self._memories[int(nb_id)].load_state_dict(nb_state)

        if 'recent_scores' in state:
            for nb_id, scores in state['recent_scores'].items():
                self._recent_scores[int(nb_id)] = deque(scores, maxlen=5000)


# =============================================================================
# Legacy Alias (for compatibility with existing code)
# =============================================================================

NeighborhoodMemoryModule = MemoryModule


__all__ = [
    'MemoryModule',
    'NeighborhoodMemory',
    'NeighborhoodMemoryModule',
    'NEIGHBORHOODS',
    'NUM_NEIGHBORHOODS',
    'NEIGHBORHOOD_TO_ID',
    'ID_TO_NEIGHBORHOOD',
    'DEFAULT_MEMORY_SLOTS',
    'DEFAULT_OUT_DIM',
    'DEFAULT_K_NEIGHBORS',
    'DEFAULT_GAMMA',
]
