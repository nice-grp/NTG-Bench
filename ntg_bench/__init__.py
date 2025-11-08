"""
NTG-Bench: Evaluation framework for synthetic network traffic generators.

This package exposes the public configuration and runner APIs so end users can
integrate the benchmark into their own automation or invoke it via the CLI.
"""

from .config import BenchmarkConfig
from .benchmark import BenchmarkRunner

__all__ = ["BenchmarkConfig", "BenchmarkRunner"]
