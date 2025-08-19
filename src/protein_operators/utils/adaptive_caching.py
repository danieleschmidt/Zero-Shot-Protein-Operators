"""
Adaptive caching system for protein design operations.

Features:
- Multi-level cache hierarchy
- Intelligent cache replacement policies
- Predictive pre-loading
- Memory-efficient storage
- Performance optimization
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch

import time
import threading
import pickle
import hashlib
import gzip
import json
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, OrderedDict
from pathlib import Path
import psutil
import weakref
from functools import wraps
import asyncio

from .advanced_logger import AdvancedLogger


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"  # Fast in-memory cache
    L2_COMPRESSED = "l2_compressed"  # Compressed in-memory cache
    L3_DISK = "l3_disk"  # Disk-based cache
    L4_DISTRIBUTED = "l4_distributed"  # Distributed cache


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ADAPTIVE = "adaptive"  # Adaptive Replacement Cache (ARC)
    PREDICTIVE = "predictive"  # Predictive eviction


class CacheHitType(Enum):
    """Types of cache hits."""
    HIT = "hit"
    MISS = "miss"
    PARTIAL_HIT = "partial_hit"


@dataclass
class CacheEntry:
    """Entry in the cache."""
    key: str
    value: Any
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    access_frequency: float = 0.0
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl: Optional[float] = None  # Time to live


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    avg_access_time: float = 0.0
    hit_ratio: float = 0.0
    memory_efficiency: float = 0.0


class AccessPatternPredictor:
    """
    Predicts future access patterns for intelligent pre-loading.
    
    Uses simple heuristics and pattern recognition to predict
    which items are likely to be accessed soon.
    """
    
    def __init__(self, history_size: int = 1000):
        self.access_history = deque(maxlen=history_size)
        self.access_patterns = defaultdict(list)
        self.sequence_patterns = defaultdict(int)
        self.temporal_patterns = defaultdict(list)
        self.lock = threading.Lock()
        
    def record_access(self, key: str, timestamp: Optional[float] = None):
        """Record an access event."""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            self.access_history.append((key, timestamp))
            
            # Update access patterns
            self.access_patterns[key].append(timestamp)
            
            # Update sequence patterns
            if len(self.access_history) >= 2:
                prev_key = self.access_history[-2][0]
                sequence = f"{prev_key}->{key}"
                self.sequence_patterns[sequence] += 1
            
            # Update temporal patterns
            self.temporal_patterns[key].append(timestamp)
            
            # Keep only recent patterns
            if len(self.access_patterns[key]) > 100:
                self.access_patterns[key] = self.access_patterns[key][-100:]
            
            if len(self.temporal_patterns[key]) > 50:
                self.temporal_patterns[key] = self.temporal_patterns[key][-50:]
    
    def predict_next_accesses(self, current_key: str, n_predictions: int = 5) -> List[Tuple[str, float]]:
        """Predict next likely accesses with confidence scores."""
        predictions = []
        
        with self.lock:
            # Sequence-based predictions
            sequence_predictions = []
            for sequence, count in self.sequence_patterns.items():
                if sequence.startswith(f"{current_key}->"):
                    next_key = sequence.split("->")[1]
                    confidence = count / sum(self.sequence_patterns.values())
                    sequence_predictions.append((next_key, confidence))
            
            # Sort by confidence and take top predictions
            sequence_predictions.sort(key=lambda x: x[1], reverse=True)
            predictions.extend(sequence_predictions[:n_predictions])
            
            # Temporal pattern predictions
            current_time = time.time()
            temporal_predictions = []
            
            for key, timestamps in self.temporal_patterns.items():
                if len(timestamps) >= 3:
                    # Calculate average interval
                    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                    avg_interval = sum(intervals) / len(intervals)
                    
                    # Predict next access time
                    last_access = timestamps[-1]
                    predicted_next = last_access + avg_interval
                    
                    # Calculate confidence based on interval consistency
                    interval_variance = sum((interval - avg_interval) ** 2 for interval in intervals) / len(intervals)
                    confidence = 1.0 / (1.0 + interval_variance)
                    
                    # Adjust confidence based on time proximity
                    time_factor = max(0, 1.0 - abs(predicted_next - current_time) / 3600)  # 1 hour window
                    confidence *= time_factor
                    
                    if confidence > 0.1:  # Minimum confidence threshold
                        temporal_predictions.append((key, confidence))
            
            temporal_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Combine predictions and remove duplicates
            all_predictions = {}
            for key, conf in sequence_predictions + temporal_predictions:
                if key in all_predictions:
                    all_predictions[key] = max(all_predictions[key], conf)
                else:
                    all_predictions[key] = conf
            
            # Sort and return top predictions
            final_predictions = list(all_predictions.items())
            final_predictions.sort(key=lambda x: x[1], reverse=True)
            
            return final_predictions[:n_predictions]
    
    def get_access_frequency(self, key: str) -> float:
        """Get access frequency for a key."""
        with self.lock:
            if key not in self.access_patterns:
                return 0.0
            
            timestamps = self.access_patterns[key]
            if len(timestamps) < 2:
                return 0.0
            
            time_span = timestamps[-1] - timestamps[0]
            if time_span == 0:
                return float('inf')
            
            return len(timestamps) / time_span  # accesses per second


class AdaptiveReplacementCache:
    """
    Adaptive Replacement Cache (ARC) implementation.
    
    Balances between recency and frequency by maintaining
    multiple LRU lists with adaptive partitioning.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.p = 0  # Adaptation parameter
        
        # Four LRU lists
        self.t1 = OrderedDict()  # Recent items (not in cache)
        self.t2 = OrderedDict()  # Frequent items (not in cache)
        self.b1 = OrderedDict()  # Recently evicted from t1
        self.b2 = OrderedDict()  # Recently evicted from t2
        
        self.cache = {}  # Actual cache storage
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Cache hit - move to MRU position
                if key in self.t1:
                    self.t1.pop(key)
                    self.t2[key] = True
                elif key in self.t2:
                    self.t2.move_to_end(key)
                
                return self.cache[key]
            
            return None
    
    def put(self, key: str, value: Any, size: int = 1):
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing item
                if key in self.t1:
                    self.t1.move_to_end(key)
                elif key in self.t2:
                    self.t2.move_to_end(key)
                self.cache[key] = value
                return
            
            # Cache miss - check ghost lists
            if key in self.b1:
                # Item was recently evicted from t1
                self.p = min(self.capacity, self.p + max(1, len(self.b2) // len(self.b1)))
                self.b1.pop(key)
                self._replace(key)
                self.t2[key] = True
                self.cache[key] = value
            
            elif key in self.b2:
                # Item was recently evicted from t2
                self.p = max(0, self.p - max(1, len(self.b1) // len(self.b2)))
                self.b2.pop(key)
                self._replace(key)
                self.t2[key] = True
                self.cache[key] = value
            
            else:
                # New item
                if len(self.t1) + len(self.b1) == self.capacity:
                    if len(self.t1) < self.capacity:
                        # Delete LRU from b1
                        self.b1.popitem(last=False)
                        self._replace(key)
                    else:
                        # Delete LRU from t1
                        lru_key = next(iter(self.t1))
                        self.t1.pop(lru_key)
                        del self.cache[lru_key]
                
                elif len(self.t1) + len(self.b1) < self.capacity:
                    total_cache = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
                    if total_cache >= self.capacity:
                        if total_cache == 2 * self.capacity:
                            # Delete LRU from b2
                            self.b2.popitem(last=False)
                        self._replace(key)
                
                self.t1[key] = True
                self.cache[key] = value
    
    def _replace(self, key: str):
        """Replace an item to make room."""
        if len(self.t1) > 0 and (len(self.t1) > self.p or (key in self.b2 and len(self.t1) == self.p)):
            # Move LRU from t1 to b1
            lru_key = next(iter(self.t1))
            self.t1.pop(lru_key)
            self.b1[lru_key] = True
            del self.cache[lru_key]
        else:
            # Move LRU from t2 to b2
            if len(self.t2) > 0:
                lru_key = next(iter(self.t2))
                self.t2.pop(lru_key)
                self.b2[lru_key] = True
                del self.cache[lru_key]
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.t1:
                    self.t1.pop(key)
                elif key in self.t2:
                    self.t2.pop(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.t1.clear()
            self.t2.clear()
            self.b1.clear()
            self.b2.clear()
            self.p = 0
    
    def size(self) -> int:
        """Get number of items in cache."""
        return len(self.cache)


class CompressionManager:
    """
    Manages compression and decompression of cache entries.
    
    Uses different compression algorithms based on data type
    and size to optimize memory usage.
    """
    
    def __init__(self):
        self.compression_stats = defaultdict(list)
        self.logger = AdvancedLogger(__name__)
    
    def compress(self, data: Any, compression_level: int = 6) -> Tuple[bytes, float]:
        """Compress data and return compressed bytes with compression ratio."""
        try:
            # Serialize data
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            # Compress
            compressed = gzip.compress(serialized, compresslevel=compression_level)
            compressed_size = len(compressed)
            
            # Calculate compression ratio
            ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            # Record stats
            self.compression_stats['ratios'].append(ratio)
            self.compression_stats['original_sizes'].append(original_size)
            self.compression_stats['compressed_sizes'].append(compressed_size)
            
            return compressed, ratio
        
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            # Fallback to uncompressed
            serialized = pickle.dumps(data)
            return serialized, 1.0
    
    def decompress(self, compressed_data: bytes) -> Any:
        """Decompress data and return original object."""
        try:
            # Try gzip decompression first
            decompressed = gzip.decompress(compressed_data)
            return pickle.loads(decompressed)
        
        except Exception:
            # Fallback to direct pickle loading (for uncompressed data)
            try:
                return pickle.loads(compressed_data)
            except Exception as e:
                self.logger.error(f"Decompression failed: {e}")
                raise
    
    def should_compress(self, data_size: int, data_type: type) -> bool:
        """Determine if data should be compressed based on size and type."""
        # Don't compress small data
        if data_size < 1024:  # 1KB threshold
            return False
        
        # Always compress large data
        if data_size > 1024 * 1024:  # 1MB threshold
            return True
        
        # Compress specific types that typically compress well
        compressible_types = (str, list, dict, torch.Tensor if hasattr(torch, 'Tensor') else type(None))
        
        if isinstance(data_type, type) and issubclass(data_type, compressible_types):
            return True
        
        return False
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics."""
        if not self.compression_stats['ratios']:
            return {'avg_ratio': 1.0, 'total_saved_bytes': 0.0}
        
        ratios = self.compression_stats['ratios']
        original_sizes = self.compression_stats['original_sizes']
        compressed_sizes = self.compression_stats['compressed_sizes']
        
        avg_ratio = sum(ratios) / len(ratios)
        total_saved = sum(original_sizes) - sum(compressed_sizes)
        
        return {
            'avg_ratio': avg_ratio,
            'total_saved_bytes': total_saved,
            'compression_count': len(ratios)
        }


class AdaptiveCacheSystem:
    """
    Multi-level adaptive caching system for protein design operations.
    
    Provides intelligent caching with multiple levels, adaptive policies,
    and predictive pre-loading for optimal performance.
    """
    
    def __init__(
        self,
        l1_size_mb: int = 256,
        l2_size_mb: int = 1024,
        l3_size_mb: int = 4096,
        cache_dir: Optional[str] = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
        enable_compression: bool = True,
        enable_prediction: bool = True
    ):
        self.logger = AdvancedLogger(__name__)
        
        # Configuration
        self.l1_size_bytes = l1_size_mb * 1024 * 1024
        self.l2_size_bytes = l2_size_mb * 1024 * 1024
        self.l3_size_bytes = l3_size_mb * 1024 * 1024
        
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression
        self.enable_prediction = enable_prediction
        
        # Initialize cache levels
        self._init_cache_levels()
        
        # Components
        if enable_compression:
            self.compression_manager = CompressionManager()
        else:
            self.compression_manager = None
        
        if enable_prediction:
            self.access_predictor = AccessPatternPredictor()
        else:
            self.access_predictor = None
        
        # Statistics and monitoring
        self.stats = {level: CacheStats() for level in CacheLevel}
        self.global_stats = CacheStats()
        
        # Threading
        self.lock = threading.RLock()
        self.background_thread = None
        self.running = False
        
        # Performance tracking
        self.access_times = deque(maxlen=1000)
        self.hit_rates = deque(maxlen=100)
        
        self.logger.info("Adaptive Cache System initialized")
    
    def _init_cache_levels(self):
        """Initialize cache levels based on eviction policy."""
        if self.eviction_policy == EvictionPolicy.ADAPTIVE:
            # Use ARC for L1 cache
            l1_capacity = self.l1_size_bytes // 1024  # Estimate entries
            self.l1_cache = AdaptiveReplacementCache(l1_capacity)
        else:
            # Use OrderedDict for LRU
            self.l1_cache = OrderedDict()
        
        self.l2_cache = OrderedDict()  # Compressed cache
        self.l3_cache = {}  # File-based cache metadata
        
        # Current sizes
        self.l1_current_size = 0
        self.l2_current_size = 0
        self.l3_current_size = 0
    
    def start_background_tasks(self):
        """Start background maintenance tasks."""
        if self.running:
            return
        
        self.running = True
        self.background_thread = threading.Thread(
            target=self._background_maintenance,
            daemon=True
        )
        self.background_thread.start()
        
        self.logger.info("Background cache maintenance started")
    
    def stop_background_tasks(self):
        """Stop background maintenance tasks."""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=5.0)
        
        self.logger.info("Background cache maintenance stopped")
    
    def get(self, key: str) -> Tuple[Optional[Any], CacheHitType]:
        """Get item from cache with hit type information."""
        start_time = time.time()
        
        with self.lock:
            # Record access for prediction
            if self.access_predictor:
                self.access_predictor.record_access(key)
            
            # Try L1 cache first
            if self.eviction_policy == EvictionPolicy.ADAPTIVE:
                value = self.l1_cache.get(key)
                if value is not None:
                    self._record_hit(CacheLevel.L1_MEMORY, time.time() - start_time)
                    return value, CacheHitType.HIT
            else:
                if key in self.l1_cache:
                    # Move to end (LRU)
                    value = self.l1_cache.pop(key)
                    self.l1_cache[key] = value
                    self._record_hit(CacheLevel.L1_MEMORY, time.time() - start_time)
                    return value, CacheHitType.HIT
            
            # Try L2 cache (compressed)
            if key in self.l2_cache:
                compressed_entry = self.l2_cache.pop(key)
                self.l2_cache[key] = compressed_entry  # Move to end
                
                # Decompress
                if self.compression_manager:
                    try:
                        value = self.compression_manager.decompress(compressed_entry)
                        
                        # Promote to L1 if there's space
                        self._promote_to_l1(key, value)
                        
                        self._record_hit(CacheLevel.L2_COMPRESSED, time.time() - start_time)
                        return value, CacheHitType.HIT
                    
                    except Exception as e:
                        self.logger.error(f"L2 decompression failed for {key}: {e}")
                        # Remove corrupted entry
                        del self.l2_cache[key]
            
            # Try L3 cache (disk)
            if key in self.l3_cache:
                try:
                    file_path = self.l3_cache[key]['file_path']
                    if Path(file_path).exists():
                        with open(file_path, 'rb') as f:
                            if self.compression_manager:
                                compressed_data = f.read()
                                value = self.compression_manager.decompress(compressed_data)
                            else:
                                value = pickle.load(f)
                        
                        # Promote to higher levels
                        self._promote_to_l2(key, value)
                        self._promote_to_l1(key, value)
                        
                        self._record_hit(CacheLevel.L3_DISK, time.time() - start_time)
                        return value, CacheHitType.HIT
                    else:
                        # File missing, remove from metadata
                        del self.l3_cache[key]
                
                except Exception as e:
                    self.logger.error(f"L3 cache read failed for {key}: {e}")
                    if key in self.l3_cache:
                        del self.l3_cache[key]
            
            # Cache miss
            self._record_miss(time.time() - start_time)
            return None, CacheHitType.MISS
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put item in cache with optional TTL."""
        with self.lock:
            # Calculate size
            try:
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
            except Exception:
                self.logger.error(f"Cannot serialize value for key {key}")
                return
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            # Determine which levels to store in
            self._store_in_appropriate_levels(entry)
            
            # Update statistics
            self.global_stats.entry_count += 1
            self.global_stats.size_bytes += size_bytes
    
    def _store_in_appropriate_levels(self, entry: CacheEntry):
        """Store entry in appropriate cache levels."""
        key = entry.key
        value = entry.value
        size_bytes = entry.size_bytes
        
        # Always try to store in L1 first
        if self._can_fit_in_l1(size_bytes):
            if self.eviction_policy == EvictionPolicy.ADAPTIVE:
                self.l1_cache.put(key, value, size_bytes)
            else:
                # Make room if needed
                self._make_room_l1(size_bytes)
                self.l1_cache[key] = value
            
            self.l1_current_size += size_bytes
            self.stats[CacheLevel.L1_MEMORY].entry_count += 1
            self.stats[CacheLevel.L1_MEMORY].size_bytes += size_bytes
        
        # Store in L2 if compression is enabled and beneficial
        elif self.enable_compression and self.compression_manager:
            if self.compression_manager.should_compress(size_bytes, type(value)):
                compressed_data, ratio = self.compression_manager.compress(value)
                compressed_size = len(compressed_data)
                
                if self._can_fit_in_l2(compressed_size):
                    # Make room if needed
                    self._make_room_l2(compressed_size)
                    
                    self.l2_cache[key] = compressed_data
                    self.l2_current_size += compressed_size
                    
                    self.stats[CacheLevel.L2_COMPRESSED].entry_count += 1
                    self.stats[CacheLevel.L2_COMPRESSED].size_bytes += compressed_size
                    
                    entry.compression_ratio = ratio
                else:
                    # Store in L3
                    self._store_in_l3(key, value, entry)
            else:
                # Store in L3 without compression
                self._store_in_l3(key, value, entry)
        
        else:
            # Store in L3
            self._store_in_l3(key, value, entry)
    
    def _store_in_l3(self, key: str, value: Any, entry: CacheEntry):
        """Store entry in L3 disk cache."""
        try:
            # Create filename based on key hash
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            file_path = self.cache_dir / f"cache_{key_hash}.pkl"
            
            # Serialize and optionally compress
            if self.compression_manager:
                compressed_data, ratio = self.compression_manager.compress(value)
                with open(file_path, 'wb') as f:
                    f.write(compressed_data)
                size_on_disk = len(compressed_data)
                entry.compression_ratio = ratio
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                size_on_disk = file_path.stat().st_size
            
            # Update metadata
            self.l3_cache[key] = {
                'file_path': str(file_path),
                'size_bytes': size_on_disk,
                'created_at': time.time(),
                'ttl': entry.ttl
            }
            
            self.l3_current_size += size_on_disk
            self.stats[CacheLevel.L3_DISK].entry_count += 1
            self.stats[CacheLevel.L3_DISK].size_bytes += size_on_disk
            
            # Check L3 size limit
            if self.l3_current_size > self.l3_size_bytes:
                self._evict_from_l3()
        
        except Exception as e:
            self.logger.error(f"Failed to store {key} in L3 cache: {e}")
    
    def _can_fit_in_l1(self, size_bytes: int) -> bool:
        """Check if item can fit in L1 cache."""
        return (self.l1_current_size + size_bytes) <= self.l1_size_bytes
    
    def _can_fit_in_l2(self, size_bytes: int) -> bool:
        """Check if item can fit in L2 cache."""
        return (self.l2_current_size + size_bytes) <= self.l2_size_bytes
    
    def _make_room_l1(self, needed_bytes: int):
        """Make room in L1 cache."""
        if self.eviction_policy == EvictionPolicy.ADAPTIVE:
            # ARC handles its own eviction
            return
        
        while self.l1_current_size + needed_bytes > self.l1_size_bytes and self.l1_cache:
            # Evict LRU item
            key, value = self.l1_cache.popitem(last=False)
            
            # Estimate size (simplified)
            try:
                size_bytes = len(pickle.dumps(value))
                self.l1_current_size -= size_bytes
            except Exception:
                self.l1_current_size = max(0, self.l1_current_size - 1024)  # Estimate
            
            # Try to promote to L2
            self._promote_to_l2(key, value)
            
            self.stats[CacheLevel.L1_MEMORY].evictions += 1
    
    def _make_room_l2(self, needed_bytes: int):
        """Make room in L2 cache."""
        while self.l2_current_size + needed_bytes > self.l2_size_bytes and self.l2_cache:
            # Evict LRU item
            key, compressed_data = self.l2_cache.popitem(last=False)
            
            size_bytes = len(compressed_data)
            self.l2_current_size -= size_bytes
            
            # Try to promote to L3
            if self.compression_manager:
                try:
                    value = self.compression_manager.decompress(compressed_data)
                    entry = CacheEntry(key=key, value=value, size_bytes=size_bytes)
                    self._store_in_l3(key, value, entry)
                except Exception as e:
                    self.logger.error(f"Failed to decompress {key} for L3 promotion: {e}")
            
            self.stats[CacheLevel.L2_COMPRESSED].evictions += 1
    
    def _evict_from_l3(self):
        """Evict items from L3 cache to stay within size limit."""
        # Sort by access time (LRU)
        l3_items = [(k, v) for k, v in self.l3_cache.items()]
        l3_items.sort(key=lambda x: x[1]['created_at'])
        
        # Evict oldest items until we're under the limit
        while self.l3_current_size > self.l3_size_bytes and l3_items:
            key, metadata = l3_items.pop(0)
            
            # Remove file
            try:
                Path(metadata['file_path']).unlink(missing_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to remove L3 cache file: {e}")
            
            # Update size and remove from metadata
            self.l3_current_size -= metadata['size_bytes']
            del self.l3_cache[key]
            
            self.stats[CacheLevel.L3_DISK].evictions += 1
    
    def _promote_to_l1(self, key: str, value: Any):
        """Promote item to L1 cache if there's room."""
        try:
            size_bytes = len(pickle.dumps(value))
            if self._can_fit_in_l1(size_bytes):
                if self.eviction_policy == EvictionPolicy.ADAPTIVE:
                    self.l1_cache.put(key, value, size_bytes)
                else:
                    self.l1_cache[key] = value
                
                self.l1_current_size += size_bytes
                self.stats[CacheLevel.L1_MEMORY].entry_count += 1
                self.stats[CacheLevel.L1_MEMORY].size_bytes += size_bytes
        except Exception:
            pass  # Silently fail promotion
    
    def _promote_to_l2(self, key: str, value: Any):
        """Promote item to L2 cache if compression is beneficial."""
        if not self.enable_compression or not self.compression_manager:
            return
        
        try:
            size_bytes = len(pickle.dumps(value))
            if self.compression_manager.should_compress(size_bytes, type(value)):
                compressed_data, ratio = self.compression_manager.compress(value)
                compressed_size = len(compressed_data)
                
                if self._can_fit_in_l2(compressed_size):
                    self.l2_cache[key] = compressed_data
                    self.l2_current_size += compressed_size
                    
                    self.stats[CacheLevel.L2_COMPRESSED].entry_count += 1
                    self.stats[CacheLevel.L2_COMPRESSED].size_bytes += compressed_size
        except Exception:
            pass  # Silently fail promotion
    
    def _record_hit(self, level: CacheLevel, access_time: float):
        """Record cache hit statistics."""
        self.stats[level].hits += 1
        self.global_stats.hits += 1
        
        self.access_times.append(access_time)
        
        # Update hit ratio
        total_requests = self.global_stats.hits + self.global_stats.misses
        if total_requests > 0:
            self.global_stats.hit_ratio = self.global_stats.hits / total_requests
            self.hit_rates.append(self.global_stats.hit_ratio)
    
    def _record_miss(self, access_time: float):
        """Record cache miss statistics."""
        self.global_stats.misses += 1
        self.access_times.append(access_time)
        
        # Update hit ratio
        total_requests = self.global_stats.hits + self.global_stats.misses
        if total_requests > 0:
            self.global_stats.hit_ratio = self.global_stats.hits / total_requests
            self.hit_rates.append(self.global_stats.hit_ratio)
    
    def _background_maintenance(self):
        """Background maintenance tasks."""
        while self.running:
            try:
                with self.lock:
                    # Clean expired entries
                    self._clean_expired_entries()
                    
                    # Update access frequency statistics
                    self._update_access_frequencies()
                    
                    # Predictive pre-loading
                    if self.enable_prediction:
                        self._predictive_preload()
                    
                    # Update performance metrics
                    self._update_performance_metrics()
                
                time.sleep(60)  # Run every minute
            
            except Exception as e:
                self.logger.error(f"Background maintenance error: {e}")
                time.sleep(60)
    
    def _clean_expired_entries(self):
        """Remove expired cache entries."""
        current_time = time.time()
        
        # Clean L3 cache
        expired_keys = []
        for key, metadata in self.l3_cache.items():
            if metadata.get('ttl') and (current_time - metadata['created_at']) > metadata['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            metadata = self.l3_cache.pop(key)
            try:
                Path(metadata['file_path']).unlink(missing_ok=True)
                self.l3_current_size -= metadata['size_bytes']
            except Exception as e:
                self.logger.error(f"Failed to remove expired L3 entry {key}: {e}")
    
    def _update_access_frequencies(self):
        """Update access frequency statistics."""
        if not self.access_predictor:
            return
        
        # This would update frequency-based eviction policies
        pass
    
    def _predictive_preload(self):
        """Predictively preload likely-to-be-accessed items."""
        if not self.access_predictor:
            return
        
        # Get recent access pattern
        if not self.access_predictor.access_history:
            return
        
        recent_key = self.access_predictor.access_history[-1][0]
        predictions = self.access_predictor.predict_next_accesses(recent_key, 3)
        
        for predicted_key, confidence in predictions:
            if confidence > 0.5:  # High confidence threshold
                # Try to promote from lower levels
                if predicted_key in self.l3_cache and predicted_key not in self.l2_cache and predicted_key not in self.l1_cache:
                    # Preload from L3 to L2
                    try:
                        metadata = self.l3_cache[predicted_key]
                        file_path = metadata['file_path']
                        if Path(file_path).exists():
                            with open(file_path, 'rb') as f:
                                if self.compression_manager:
                                    compressed_data = f.read()
                                    value = self.compression_manager.decompress(compressed_data)
                                else:
                                    value = pickle.load(f)
                            
                            self._promote_to_l2(predicted_key, value)
                    
                    except Exception as e:
                        self.logger.debug(f"Predictive preload failed for {predicted_key}: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        if self.access_times:
            self.global_stats.avg_access_time = sum(self.access_times) / len(self.access_times)
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry across all levels."""
        with self.lock:
            found = False
            
            # Remove from L1
            if self.eviction_policy == EvictionPolicy.ADAPTIVE:
                if self.l1_cache.remove(key):
                    found = True
            else:
                if key in self.l1_cache:
                    del self.l1_cache[key]
                    found = True
            
            # Remove from L2
            if key in self.l2_cache:
                del self.l2_cache[key]
                found = True
            
            # Remove from L3
            if key in self.l3_cache:
                metadata = self.l3_cache.pop(key)
                try:
                    Path(metadata['file_path']).unlink(missing_ok=True)
                    self.l3_current_size -= metadata['size_bytes']
                except Exception as e:
                    self.logger.error(f"Failed to remove L3 file for {key}: {e}")
                found = True
            
            return found
    
    def clear_all(self):
        """Clear all cache levels."""
        with self.lock:
            # Clear L1
            if self.eviction_policy == EvictionPolicy.ADAPTIVE:
                self.l1_cache.clear()
            else:
                self.l1_cache.clear()
            self.l1_current_size = 0
            
            # Clear L2
            self.l2_cache.clear()
            self.l2_current_size = 0
            
            # Clear L3
            for metadata in self.l3_cache.values():
                try:
                    Path(metadata['file_path']).unlink(missing_ok=True)
                except Exception:
                    pass
            
            self.l3_cache.clear()
            self.l3_current_size = 0
            
            # Reset statistics
            for stats in self.stats.values():
                stats.hits = 0
                stats.misses = 0
                stats.evictions = 0
                stats.size_bytes = 0
                stats.entry_count = 0
            
            self.global_stats = CacheStats()
        
        self.logger.info("All cache levels cleared")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self.lock:
            report = {
                'global_stats': {
                    'hit_ratio': self.global_stats.hit_ratio,
                    'total_hits': self.global_stats.hits,
                    'total_misses': self.global_stats.misses,
                    'avg_access_time_ms': self.global_stats.avg_access_time * 1000,
                    'total_entries': sum(stats.entry_count for stats in self.stats.values()),
                    'total_size_mb': sum(stats.size_bytes for stats in self.stats.values()) / (1024 * 1024)
                },
                'level_stats': {},
                'memory_usage': {
                    'l1_usage_mb': self.l1_current_size / (1024 * 1024),
                    'l1_capacity_mb': self.l1_size_bytes / (1024 * 1024),
                    'l2_usage_mb': self.l2_current_size / (1024 * 1024),
                    'l2_capacity_mb': self.l2_size_bytes / (1024 * 1024),
                    'l3_usage_mb': self.l3_current_size / (1024 * 1024),
                    'l3_capacity_mb': self.l3_size_bytes / (1024 * 1024)
                }
            }
            
            # Level-specific statistics
            for level, stats in self.stats.items():
                total_requests = stats.hits + stats.misses
                hit_ratio = stats.hits / total_requests if total_requests > 0 else 0.0
                
                report['level_stats'][level.value] = {
                    'hits': stats.hits,
                    'misses': stats.misses,
                    'evictions': stats.evictions,
                    'hit_ratio': hit_ratio,
                    'entries': stats.entry_count,
                    'size_mb': stats.size_bytes / (1024 * 1024)
                }
            
            # Compression statistics
            if self.compression_manager:
                compression_stats = self.compression_manager.get_compression_stats()
                report['compression'] = compression_stats
            
            # Recent performance trends
            if self.hit_rates:
                report['trends'] = {
                    'recent_hit_rates': list(self.hit_rates)[-10:],  # Last 10 measurements
                    'avg_recent_hit_rate': sum(list(self.hit_rates)[-10:]) / min(10, len(self.hit_rates))
                }
        
        return report


def cached(cache_system: AdaptiveCacheSystem, ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache_system: The cache system to use
        ttl: Time to live for cached results
        key_func: Function to generate cache key from arguments
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "|".join(key_parts)
            
            # Try to get from cache
            result, hit_type = cache_system.get(cache_key)
            
            if hit_type == CacheHitType.HIT:
                return result
            
            # Cache miss - compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_system.put(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator
