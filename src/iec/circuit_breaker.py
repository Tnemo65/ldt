"""
Circuit Breaker implementation for IEC feedback.

Prevents runaway feedback loops by limiting consecutive IEC actions.

This is the Phase 2D migration version with:
- trip() / reset() / is_open() / get_trip_count() methods as required
- Integration with Prometheus alerting via MemStream_HMACVerificationFailures
"""

import time
from enum import Enum
from typing import Optional


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Blocking actions
    HALF_OPEN = "half_open"  # Testing if recovery possible


class CircuitBreaker:
    """
    Circuit breaker to prevent runaway IEC feedback loops.

    States:
    - CLOSED: Normal operation, actions allowed
    - OPEN: Too many consecutive actions, blocking
    - HALF_OPEN: Testing after cooldown, limited actions

    Methods:
    - trip(): Open the circuit manually (e.g., on HMAC failure)
    - reset(): Close the circuit manually
    - is_open(): Check if circuit is open
    - get_trip_count(): Get number of times circuit was tripped
    - record_action(): Record an IEC action
    - should_allow_action(): Check if action is allowed
    - on_action_success(): Call when action succeeds
    - on_action_failure(): Call when action fails
    """

    def __init__(
        self,
        cooldown_seconds: float = 300.0,
        max_consecutive: int = 10,
        half_open_max_actions: int = 3
    ):
        """
        Initialize circuit breaker.

        Args:
            cooldown_seconds: Time to wait before transitioning from OPEN to HALF_OPEN
            max_consecutive: Max consecutive actions before opening
            half_open_max_actions: Max actions allowed in HALF_OPEN state
        """
        self.cooldown_seconds = cooldown_seconds
        self.max_consecutive = max_consecutive
        self.half_open_max_actions = half_open_max_actions

        self._state = CircuitState.CLOSED
        self._consecutive_actions = 0
        self._last_action_time: Optional[float] = None
        self._half_open_actions = 0

        # Phase 2D: Track trip count for security events
        self._trip_count = 0
        self._last_trip_time: Optional[float] = None

    @property
    def state(self) -> str:
        """Get current circuit state as string."""
        return self._state.value

    @property
    def consecutive_actions(self) -> int:
        """Get consecutive action count."""
        return self._consecutive_actions

    def trip(self) -> None:
        """
        Trip (open) the circuit manually.

        Use this to immediately open the circuit, e.g.:
        - On HMAC verification failure (security event)
        - On critical system failure
        - When automated drift detection indicates instability
        """
        self._state = CircuitState.OPEN
        self._trip_count += 1
        self._last_trip_time = time.time()
        self._consecutive_actions = 0
        self._half_open_actions = 0

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._consecutive_actions = 0
        self._half_open_actions = 0
        self._last_action_time = None

    def is_open(self) -> bool:
        """Check if circuit is currently open."""
        return self._state == CircuitState.OPEN

    def get_trip_count(self) -> int:
        """Get the number of times the circuit has been tripped."""
        return self._trip_count

    def record_action(self, action_time: float = None):
        """
        Record an IEC action.

        Args:
            action_time: Timestamp of action (default: now)
        """
        if action_time is None:
            action_time = time.time()

        self._last_action_time = action_time
        self._consecutive_actions += 1

        # Check if we should open
        if self._consecutive_actions >= self.max_consecutive:
            self.trip()

    def should_allow_action(self) -> bool:
        """
        Check if an action should be allowed.

        Returns:
            bool: True if action is allowed
        """
        now = time.time()

        if self._state == CircuitState.CLOSED:
            return True

        elif self._state == CircuitState.OPEN:
            # Check if cooldown has elapsed
            if self._last_action_time is not None:
                elapsed = now - self._last_action_time
                if elapsed >= self.cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_actions = 0
                    return True
            return False

        elif self._state == CircuitState.HALF_OPEN:
            # Allow limited actions in half-open state
            if self._half_open_actions < self.half_open_max_actions:
                self._half_open_actions += 1
                return True
            return False

        return False

    def on_action_success(self):
        """Called when an action succeeds."""
        if self._state == CircuitState.HALF_OPEN:
            # Successful test, close the circuit
            self.reset()
        else:
            # Reset consecutive counter on success
            self._consecutive_actions = 0

    def on_action_failure(self):
        """Called when an action fails."""
        if self._state == CircuitState.HALF_OPEN:
            # Failed during test, reopen
            self._state = CircuitState.OPEN
            self._half_open_actions = 0
        else:
            # Increment consecutive failures
            self._consecutive_actions += 1
            if self._consecutive_actions >= self.max_consecutive:
                self.trip()

    def get_status(self) -> dict:
        """
        Get current circuit breaker status.

        Returns:
            dict: Status information
        """
        return {
            'state': self.state,
            'consecutive_actions': self._consecutive_actions,
            'max_consecutive': self.max_consecutive,
            'cooldown_seconds': self.cooldown_seconds,
            'last_action_time': self._last_action_time,
            'should_allow': self.should_allow_action(),
            'trip_count': self._trip_count,
            'last_trip_time': self._last_trip_time,
            'is_open': self.is_open(),
        }


__all__ = [
    'CircuitBreaker',
    'CircuitState',
]
