"""
Mock psutil module for testing without external dependencies.
Provides basic system monitoring interface compatibility.
"""

import time
import os


class MockProcess:
    """Mock process for memory info."""
    
    def memory_info(self):
        return MockMemoryInfo()


class MockMemoryInfo:
    """Mock memory info."""
    
    @property
    def rss(self):
        return 100 * 1024 * 1024  # 100 MB


class MockVirtualMemory:
    """Mock virtual memory stats."""
    
    @property
    def percent(self):
        return 45.0  # 45% usage
    
    @property
    def available(self):
        return 8 * 1024**3  # 8 GB available
    
    @property
    def used(self):
        return 4 * 1024**3  # 4 GB used
    
    @property
    def total(self):
        return 16 * 1024**3  # 16 GB total


class MockDiskUsage:
    """Mock disk usage stats."""
    
    @property
    def percent(self):
        return 60.0  # 60% usage
    
    @property
    def free(self):
        return 200 * 1024**3  # 200 GB free
    
    @property
    def total(self):
        return 500 * 1024**3  # 500 GB total


def cpu_percent(interval=None):
    """Mock CPU percentage."""
    return 25.0  # 25% CPU usage


def cpu_count():
    """Mock CPU count."""
    return 8  # 8 cores


def virtual_memory():
    """Mock virtual memory stats."""
    return MockVirtualMemory()


def disk_usage(path):
    """Mock disk usage stats."""
    return MockDiskUsage()


def Process(pid=None):
    """Mock process creation."""
    return MockProcess()


# Make the module look like real psutil
__version__ = "mock-5.9.0"