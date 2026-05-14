"""
Redis client utilities for MemStream.

Provides secure Redis client creation with TLS and authentication.
"""

import logging
from typing import Optional

import redis
from redis.exceptions import ConnectionError, TimeoutError

LOGGER = logging.getLogger('memstream.redis')


def create_redis_client(
    host: str = 'localhost',
    port: int = 6379,
    password: Optional[str] = None,
    db: int = 0,
    ssl: bool = True,
    socket_timeout: float = 5.0,
    socket_connect_timeout: float = 5.0,
    health_check_interval: int = 30,
) -> Optional[redis.Redis]:
    """
    Create a hardened Redis client with TLS and timeouts.

    Args:
        host: Redis host
        port: Redis port
        password: Redis password (optional)
        db: Redis database number
        ssl: Enable TLS/SSL
        socket_timeout: Socket timeout in seconds
        socket_connect_timeout: Connection timeout in seconds
        health_check_interval: Health check interval in seconds

    Returns:
        redis.Redis: Connected client, or None on failure
    """
    try:
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            ssl=ssl,
            ssl_cert_reqs='required' if ssl else None,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=True,
            health_check_interval=health_check_interval,
            decode_responses=False,
        )

        # Test connection
        client.ping()

        LOGGER.info(f"Redis client connected: {host}:{port}")
        return client

    except ConnectionError as e:
        LOGGER.warning(f"Redis connection failed: {e}")
        return None
    except TimeoutError as e:
        LOGGER.warning(f"Redis timeout: {e}")
        return None
    except Exception as e:
        LOGGER.warning(f"Redis error: {e}")
        return None


def check_redis_health(client: redis.Redis) -> dict:
    """
    Check Redis client health.

    Args:
        client: Redis client

    Returns:
        dict: Health status
    """
    try:
        start = __import__('time').time()
        client.ping()
        latency = __import__('time').time() - start

        info = client.info()
        return {
            'healthy': True,
            'latency_ms': latency * 1000,
            'connected_clients': info.get('connected_clients', 0),
            'used_memory': info.get('used_memory_human', 'unknown'),
        }
    except Exception as e:
        return {
            'healthy': False,
            'error': str(e),
        }


class RedisCache:
    """Simple Redis-backed cache with TTL support."""

    def __init__(self, client: redis.Redis, default_ttl: int = 300):
        """
        Initialize Redis cache.

        Args:
            client: Redis client
            default_ttl: Default TTL in seconds
        """
        self.client = client
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[bytes]:
        """Get value from cache."""
        try:
            return self.client.get(key)
        except Exception:
            return None

    def set(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with TTL."""
        try:
            ttl = ttl or self.default_ttl
            self.client.setex(key, ttl, value)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            self.client.delete(key)
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(self.client.exists(key))
        except Exception:
            return False
