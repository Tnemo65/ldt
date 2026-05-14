"""
Serialization utilities for MemStream state (v10: 34D + 10 neighborhoods).

Handles serialization/deserialization of:
- Memory state (34D autoencoder)
- Context-beta thresholds (80 total: 10 neighborhoods x 8 context cells)
- Per-neighborhood ADWIN state (10 instances)
- Beta overrides (per neighborhood)
"""

import io
import pickle
from typing import Dict

import torch


def serialize_memory(state: Dict) -> bytes:
    """
    Serialize memory state to bytes for checkpointing.

    Args:
        state: Dict containing:
            - memory: torch.Tensor [memory_len, out_dim]
            - mem_usage: torch.Tensor [memory_len]
            - mem_ptr: int
            - count: int
            - max_thres: float
            - eval_mode: bool

    Returns:
        bytes: Serialized state
    """
    # Move tensors to CPU for serialization
    serializable_state = {
        'memory': state['memory'].cpu().numpy(),
        'mem_usage': state['mem_usage'].cpu().numpy(),
        'mem_ptr': state['mem_ptr'],
        'count': state['count'],
        'max_thres': state.get('max_thres', 0.0),
        'eval_mode': state.get('eval_mode', True),
    }

    buf = io.BytesIO()
    torch.save(serializable_state, buf, pickle_module=pickle)
    return buf.getvalue()


def deserialize_memory(data: bytes) -> Dict:
    """
    Deserialize memory state from bytes.

    Args:
        data: Serialized state bytes

    Returns:
        Dict with deserialized state
    """
    buf = io.BytesIO(data)
    state = torch.load(buf, weights_only=True, pickle_module=pickle)

    return {
        'memory': torch.from_numpy(state['memory']),
        'mem_usage': torch.from_numpy(state['mem_usage']),
        'mem_ptr': state['mem_ptr'],
        'count': state['count'],
        'max_thres': state.get('max_thres', 0.0),
        'eval_mode': state.get('eval_mode', True),
    }


def serialize_full_checkpoint(
    memory_state: bytes,
    beta_overrides: Dict[str, float],
    metadata: Dict = None
) -> bytes:
    """
    Serialize full operator checkpoint (v10: 34D + 10 neighborhoods).

    Args:
        memory_state: Serialized memory state
        beta_overrides: Per-key beta overrides
        metadata: Additional metadata

    Returns:
        bytes: Full checkpoint
    """
    checkpoint = {
        'memory_state': memory_state,
        'beta_overrides': beta_overrides,
        'metadata': metadata or {},
        'version': 'v10',
        'num_features': 34,
        'num_neighborhoods': 10,
    }

    buf = io.BytesIO()
    torch.save(checkpoint, buf, pickle_module=pickle)
    return buf.getvalue()


def deserialize_full_checkpoint(data: bytes) -> Dict:
    """
    Deserialize full operator checkpoint.

    Args:
        data: Serialized checkpoint

    Returns:
        Dict with checkpoint components
    """
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True, pickle_module=pickle)


# =============================================================================
# Context Beta Serialization (80 thresholds)
# =============================================================================

def serialize_context_beta(context_beta) -> bytes:
    """
    Serialize ContextBeta thresholds.

    Args:
        context_beta: ContextBeta instance

    Returns:
        bytes: Serialized thresholds
    """
    checkpoint = {
        'thresholds': context_beta._thresholds.cpu().numpy(),
        'default_beta': context_beta.default_beta,
        'fitted': context_beta._fitted,
    }

    buf = io.BytesIO()
    torch.save(checkpoint, buf, pickle_module=pickle)
    return buf.getvalue()


def deserialize_context_beta(data: bytes) -> Dict:
    """
    Deserialize ContextBeta thresholds.

    Args:
        data: Serialized thresholds

    Returns:
        Dict with thresholds and metadata
    """
    from .context_aware import ContextBeta

    buf = io.BytesIO(data)
    state = torch.load(buf, weights_only=True, pickle_module=pickle)

    cb = ContextBeta(default_beta=state['default_beta'])
    cb._thresholds = torch.from_numpy(state['thresholds']).float()
    cb._fitted = state['fitted']

    return cb


# =============================================================================
# ADWIN State Serialization (10 instances)
# =============================================================================

def serialize_adwin_state(adwin_manager) -> bytes:
    """
    Serialize NeighborhoodADWINManager state (10 ADWIN instances).

    Args:
        adwin_manager: NeighborhoodADWINManager instance

    Returns:
        bytes: Serialized ADWIN state
    """
    adwin_states = {}
    for nb_id, adwin in adwin_manager._adwins.items():
        adwin_states[str(nb_id)] = {
            'window': list(adwin._window),
            'total': adwin._total,
            'n': adwin._n,
        }

    checkpoint = {
        'adwin_states': adwin_states,
        'delta': adwin_manager.delta,
        'max_window': adwin_manager.max_window,
    }

    buf = io.BytesIO()
    torch.save(checkpoint, buf, pickle_module=pickle)
    return buf.getvalue()


def deserialize_adwin_state(data: bytes):
    """
    Deserialize NeighborhoodADWINManager state.

    Args:
        data: Serialized ADWIN state

    Returns:
        NeighborhoodADWINManager with restored state
    """
    from collections import deque
    from .context_aware import NeighborhoodADWINManager, ADWIN

    buf = io.BytesIO(data)
    state = torch.load(buf, weights_only=True, pickle_module=pickle)

    manager = NeighborhoodADWINManager(
        delta=state['delta'],
        max_window=state['max_window']
    )

    for nb_id_str, adwin_state in state['adwin_states'].items():
        nb_id = int(nb_id_str)
        adwin = manager._adwins[nb_id]
        adwin._window = deque(adwin_state['window'], maxlen=adwin.max_window)
        adwin._total = adwin_state['total']
        adwin._n = adwin_state['n']

    return manager
