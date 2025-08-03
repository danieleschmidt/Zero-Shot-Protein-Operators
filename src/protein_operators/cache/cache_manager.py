"""
Cache management system for protein operators.
"""

import os
import json
import pickle
import hashlib
import logging
from typing import Any, Optional, Dict, Union, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
import threading
import time


@dataclass
class CacheConfig:
    """Configuration for cache systems."""
    cache_type: str = "memory"  # "memory", "redis", "file"
    max_size_mb: int = 1024  # Maximum cache size in MB
    ttl_seconds: int = 3600  # Time to live in seconds
    cleanup_interval: int = 300  # Cleanup interval in seconds
    
    # Redis-specific config
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # File cache config
    cache_dir: str = "cache"
    max_files: int = 10000


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear entire cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache implementation with LRU eviction."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_expired,
            daemon=True
        )
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if self._is_expired(entry):
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                self.stats['misses'] += 1
                return None
            
            # Update access time for LRU
            self.access_times[key] = time.time()
            self.stats['hits'] += 1
            
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in memory cache."""
        with self.lock:
            ttl = ttl or self.config.ttl_seconds
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            entry = {
                'value': value,
                'expires_at': expires_at,
                'created_at': datetime.now(),
                'size_bytes': self._estimate_size(value)
            }
            
            self.cache[key] = entry
            self.access_times[key] = time.time()
            self.stats['sets'] += 1
            
            # Check if eviction is needed
            self._evict_if_needed()
    
    def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                self.stats['deletes'] += 1
                return True
            return False
    
    def clear(self) -> None:
        """Clear entire memory cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self.lock:
            if key not in self.cache:
                return False
            
            entry = self.cache[key]
            if self._is_expired(entry):
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return False
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(
                entry['size_bytes'] 
                for entry in self.cache.values()
            )
            
            hit_rate = 0.0
            total_requests = self.stats['hits'] + self.stats['misses']
            if total_requests > 0:
                hit_rate = self.stats['hits'] / total_requests
            
            return {
                'backend_type': 'memory',
                'num_keys': len(self.cache),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'hit_rate': hit_rate,
                **self.stats
            }
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > entry['expires_at']
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            return len(str(value)) * 4  # Rough Unicode estimate
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache size exceeds limit."""
        total_size = sum(
            entry['size_bytes'] 
            for entry in self.cache.values()
        )
        
        max_size_bytes = self.config.max_size_mb * 1024 * 1024
        
        if total_size <= max_size_bytes:
            return
        
        # Sort by access time (LRU)
        sorted_keys = sorted(
            self.access_times.keys(),
            key=lambda k: self.access_times[k]
        )
        
        # Evict oldest entries until under limit
        for key in sorted_keys:
            if total_size <= max_size_bytes:
                break
            
            if key in self.cache:
                entry_size = self.cache[key]['size_bytes']
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                total_size -= entry_size
                self.stats['evictions'] += 1
    
    def _cleanup_expired(self) -> None:
        """Background thread to clean up expired entries."""
        while True:
            try:
                time.sleep(self.config.cleanup_interval)
                
                with self.lock:
                    expired_keys = []
                    for key, entry in self.cache.items():
                        if self._is_expired(entry):
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.cache[key]
                        if key in self.access_times:
                            del self.access_times[key]
                        
            except Exception as e:
                logging.error(f"Cache cleanup error: {e}")


class FileCache(CacheBackend):
    """File-based cache implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            self.stats['misses'] += 1
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check expiration
            if datetime.now() > data['expires_at']:
                cache_file.unlink(missing_ok=True)
                self.stats['misses'] += 1
                return None
            
            self.stats['hits'] += 1
            return data['value']
            
        except Exception as e:
            logging.error(f"Error reading cache file {cache_file}: {e}")
            cache_file.unlink(missing_ok=True)
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in file cache."""
        cache_file = self._get_cache_file(key)
        ttl = ttl or self.config.ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        data = {
            'value': value,
            'expires_at': expires_at,
            'created_at': datetime.now()
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.stats['sets'] += 1
            self._evict_if_needed()
            
        except Exception as e:
            logging.error(f"Error writing cache file {cache_file}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from file cache."""
        cache_file = self._get_cache_file(key)
        
        if cache_file.exists():
            cache_file.unlink()
            self.stats['deletes'] += 1
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear entire file cache."""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink(missing_ok=True)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            if datetime.now() > data['expires_at']:
                cache_file.unlink(missing_ok=True)
                return False
            
            return True
            
        except Exception:
            cache_file.unlink(missing_ok=True)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        hit_rate = 0.0
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests > 0:
            hit_rate = self.stats['hits'] / total_requests
        
        return {
            'backend_type': 'file',
            'num_keys': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'hit_rate': hit_rate,
            **self.stats
        }
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _evict_if_needed(self) -> None:
        """Evict files if cache exceeds limits."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        
        # Check file count limit
        if len(cache_files) > self.config.max_files:
            # Sort by modification time (LRU)
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            
            # Remove oldest files
            files_to_remove = len(cache_files) - self.config.max_files
            for cache_file in cache_files[:files_to_remove]:
                cache_file.unlink(missing_ok=True)
                self.stats['evictions'] += 1


class CacheManager:
    """Main cache manager that coordinates different cache backends."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache backend
        if config.cache_type == "memory":
            self.backend = MemoryCache(config)
        elif config.cache_type == "file":
            self.backend = FileCache(config)
        elif config.cache_type == "redis":
            try:
                from .redis_cache import RedisCache
                self.backend = RedisCache(config)
            except ImportError:
                self.logger.warning("Redis not available, falling back to memory cache")
                self.backend = MemoryCache(config)
        else:
            raise ValueError(f"Unknown cache type: {config.cache_type}")
        
        self.logger.info(f"Initialized {config.cache_type} cache backend")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            return self.backend.get(key)
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        try:
            self.backend.set(key, value, ttl)
        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            return self.backend.delete(key)
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear(self) -> None:
        """Clear entire cache."""
        try:
            self.backend.clear()
            self.logger.info("Cache cleared")
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return self.backend.exists(key)
        except Exception as e:
            self.logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            return self.backend.get_stats()
        except Exception as e:
            self.logger.error(f"Cache stats error: {e}")
            return {}
    
    def cache_result(self, cache_key: str, ttl: Optional[int] = None):
        """Decorator for caching function results."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key from function and arguments
                full_key = f"{cache_key}:{self._hash_args(args, kwargs)}"
                
                # Try to get from cache
                cached_result = self.get(full_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(full_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def _hash_args(self, args: tuple, kwargs: dict) -> str:
        """Create hash from function arguments."""
        combined = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(combined.encode()).hexdigest()[:16]


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    
    if _cache_manager is None or config is not None:
        if config is None:
            # Default configuration
            config = CacheConfig(
                cache_type=os.getenv("CACHE_TYPE", "memory"),
                max_size_mb=int(os.getenv("CACHE_MAX_SIZE_MB", "1024")),
                ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600"))
            )
        
        _cache_manager = CacheManager(config)
    
    return _cache_manager


def clear_cache() -> None:
    """Clear global cache."""
    global _cache_manager
    if _cache_manager:
        _cache_manager.clear()
        _cache_manager = None