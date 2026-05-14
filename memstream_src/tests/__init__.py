"""
Tests package for CA-DQStream + MemStream.

Contains unit tests for:
- Layer 1 operators (parse, dedup, schema, watermark)
- Layer 3 operators (voting, meta-aggregation)
- Canary rules
"""

from .test_layer1 import *
from .test_layer3 import *
from .test_canary_rules import *
