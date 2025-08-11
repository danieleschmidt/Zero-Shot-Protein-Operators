"""
Caching system for protein operators.

Provides caching mechanisms for model weights, computation results,
and frequently accessed data to improve performance.
"""

from .cache_manager import CacheManager

__all__ = [
    "CacheManager"
]