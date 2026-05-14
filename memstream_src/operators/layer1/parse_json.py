"""
ParseJsonFunction - Layer 1 JSON Parser.

Parses raw JSON bytes from Kafka into Python dicts.
Silently drops malformed records by returning None.

Reference: original_flow.md lines 376-381
"""

import json
import logging
from typing import Dict, Optional, Union

from pyflink.datastream import MapFunction

LOGGER = logging.getLogger('cadqstream.layer1')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


class ParseJsonFunction(MapFunction):
    """
    Parse raw JSON bytes to dict.
    
    Input:  byte[] (raw JSON string)
    Output: dict or None (if json.loads fails)
    
    Error handling: Silently drop malformed records (return None).
    These None records are filtered out in subsequent steps.
    """
    
    def __init__(self):
        self._parse_errors = 0
        self._total_records = 0
    
    def open(self, runtime_context):
        """Initialize counters."""
        self._parse_errors = 0
        self._total_records = 0
    
    def map(self, value: Union[bytes, str]) -> Optional[Dict]:
        """
        Parse JSON value to dict.
        
        Args:
            value: Raw JSON string (bytes or str)
            
        Returns:
            Dict if parsing succeeds, None if parsing fails.
        """
        self._total_records += 1
        
        try:
            if isinstance(value, bytes):
                decoded = value.decode('utf-8')
            elif isinstance(value, str):
                decoded = value
            else:
                decoded = str(value)
            
            parsed = json.loads(decoded)
            
            if not isinstance(parsed, dict):
                LOGGER.warning(
                    "[ParseJson] Record %d: Expected dict, got %s",
                    self._total_records, type(parsed).__name__
                )
                self._parse_errors += 1
                return None
            
            return parsed
            
        except json.JSONDecodeError as e:
            self._parse_errors += 1
            LOGGER.debug(
                "[ParseJson] JSON parse error at record %d: %s",
                self._total_records, str(e)
            )
            return None
        except UnicodeDecodeError as e:
            self._parse_errors += 1
            LOGGER.debug(
                "[ParseJson] Unicode decode error at record %d: %s",
                self._total_records, str(e)
            )
            return None
        except Exception as e:
            self._parse_errors += 1
            LOGGER.warning(
                "[ParseJson] Unexpected error at record %d: %s",
                self._total_records, str(e)
            )
            return None
    
    def get_parse_error_rate(self) -> float:
        """Get current parse error rate."""
        if self._total_records == 0:
            return 0.0
        return self._parse_errors / self._total_records


def parse_json(value: Union[bytes, str]) -> Optional[Dict]:
    """
    Standalone function to parse JSON without MapFunction context.
    
    Args:
        value: Raw JSON string (bytes or str)
        
    Returns:
        Dict if parsing succeeds, None if parsing fails.
    """
    try:
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
