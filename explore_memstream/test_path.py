#!/usr/bin/env python3
import sys, os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_MEMSTREAM_ROOT = (_SCRIPT_DIR / '..' / 'memstream_src').resolve()

print(f'DEBUG: __file__={__file__}', flush=True)
print(f'DEBUG: _SCRIPT_DIR={_SCRIPT_DIR}', flush=True)
print(f'DEBUG: _MEMSTREAM_ROOT={_MEMSTREAM_ROOT}', flush=True)
print(f'DEBUG: exists={_MEMSTREAM_ROOT.exists()}', flush=True)
print(f'DEBUG: in sys.path before={str(_MEMSTREAM_ROOT) in sys.path}', flush=True)

if str(_MEMSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMSTREAM_ROOT))
    print(f'DEBUG: inserted, sys.path[0]={sys.path[0]}', flush=True)

import argparse
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

from core.memstream_core import MemStreamCore, ContextBeta

cb = ContextBeta(n_neighborhoods=10, n_cells=8)
print(f'ContextBeta shape: {cb.betas.shape}')
