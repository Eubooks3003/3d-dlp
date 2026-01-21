"""
Timing utilities for profiling forward pass components.
Enable with TIMING_ENABLED = True or by calling enable_timing().
"""
import time
import torch
from collections import defaultdict
from contextlib import contextmanager

# Global timing state
TIMING_ENABLED = False
TIMING_SYNC_CUDA = True
_timing_stats = defaultdict(list)
_timing_depth = 0


def enable_timing(sync_cuda=True):
    """Enable timing instrumentation."""
    global TIMING_ENABLED, TIMING_SYNC_CUDA
    TIMING_ENABLED = True
    TIMING_SYNC_CUDA = sync_cuda


def disable_timing():
    """Disable timing instrumentation."""
    global TIMING_ENABLED
    TIMING_ENABLED = False


def clear_timing_stats():
    """Clear accumulated timing statistics."""
    global _timing_stats
    _timing_stats = defaultdict(list)


def get_timing_stats():
    """Get timing statistics as dict of {name: (mean, std, count)}."""
    stats = {}
    for name, times in _timing_stats.items():
        if times:
            import numpy as np
            stats[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times),
                'total': np.sum(times),
            }
    return stats


def print_timing_stats(sort_by='total'):
    """Print timing statistics in a formatted table."""
    stats = get_timing_stats()
    if not stats:
        print("[timing] No timing data collected")
        return

    # Sort by specified key
    sorted_items = sorted(stats.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)

    print("\n" + "=" * 80)
    print(f"{'Component':<40} {'Mean':>10} {'Std':>10} {'Total':>10} {'Count':>8}")
    print("=" * 80)
    for name, s in sorted_items:
        print(f"{name:<40} {s['mean']*1000:>9.2f}ms {s['std']*1000:>9.2f}ms {s['total']:>9.3f}s {s['count']:>8d}")
    print("=" * 80 + "\n")


def _sync():
    """Synchronize CUDA if enabled and available."""
    if TIMING_SYNC_CUDA and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed_section(name):
    """Context manager for timing a section of code.

    Usage:
        with timed_section("encoder"):
            output = encoder(x)
    """
    global _timing_depth
    if not TIMING_ENABLED:
        yield
        return

    # Add indentation based on nesting depth
    indent = "  " * _timing_depth
    full_name = f"{indent}{name}" if _timing_depth > 0 else name

    _sync()
    t0 = time.time()
    _timing_depth += 1
    try:
        yield
    finally:
        _timing_depth -= 1
        _sync()
        elapsed = time.time() - t0
        _timing_stats[full_name].append(elapsed)


class TimedModule:
    """Mixin class to add timing to a module's forward pass.

    Usage:
        class MyModule(nn.Module, TimedModule):
            def forward(self, x):
                with self.timed("my_operation"):
                    ...
    """
    _timing_prefix = ""

    def timed(self, name):
        """Return a timed_section context manager with module prefix."""
        full_name = f"{self._timing_prefix}/{name}" if self._timing_prefix else name
        return timed_section(full_name)

    def set_timing_prefix(self, prefix):
        """Set the prefix for timing names."""
        self._timing_prefix = prefix
