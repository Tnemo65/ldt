# =============================================================================
# Integration Module - CA-DQStream + MemStream
# =============================================================================
#
# Contains:
#   - flink_job_complete: Main Flink job integrating all 4 layers
#   - test_layer_integration: Integration tests for layers
#
# Layers:
#   Layer 1: Baseline Validation (Parse, Dedup, Schema)
#   Layer 2: Dual-Branch (Canary + MemStream)
#   Layer 3: Voting Ensemble + MetaAggregator
#   Layer 4: IEC Feedback
#
# Usage:
#   from memstream_src.integration.flink_job_complete import create_job
#
# =============================================================================

from .flink_job_complete import create_job

__all__ = [
    "create_job",
]
