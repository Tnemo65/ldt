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
# Legacy Alias (for compatibility with existing code)
# =============================================================================

# MemoryModule was removed — dead code, replaced by per-neighborhood
# buffer management in MemStreamCore. NeighborhoodMemory is still available.


__all__ = [
    'NeighborhoodMemory',
    'NEIGHBORHOODS',
    'NUM_NEIGHBORHOODS',
    'NEIGHBORHOOD_TO_ID',
    'ID_TO_NEIGHBORHOOD',
    'DEFAULT_MEMORY_SLOTS',
    'DEFAULT_OUT_DIM',
    'DEFAULT_K_NEIGHBORS',
    'DEFAULT_GAMMA',
]
