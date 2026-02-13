# Copyright 2025 Ruihang Li and DeepGen Team @ Shanghai Innovation Institute

"""
Evaluation module for DeepGen-RL.

This module provides evaluation utilities for various benchmarks:
- UniGenBench: A unified semantic evaluation benchmark for T2I generation
"""

from .unigenbench import (
    UniGenBenchScoringConfig,
    UniGenBenchScorer,
    is_unigenbench_enabled,
    parse_scoring_config,
)

__all__ = [
    "UniGenBenchScoringConfig",
    "UniGenBenchScorer",
    "is_unigenbench_enabled",
    "parse_scoring_config",
]
