# Copyright 2025 Ruihang Li and DeepGen Team @ Shanghai Innovation Institute

"""
Phase Logger for GRPO Training Process.

This module provides a PhaseLogger class that outputs phase-level logs to a file
for monitoring and debugging the GRPO training process. Only global rank 0
process writes to the log file.
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist


class PhaseLogger:
    """
    A logger for tracking phases during GRPO training.

    Only global rank 0 process writes logs to file and prints to stdout.
    Each log entry includes a full datetime stamp in YYYY-MM-DD HH:MM:SS format.

    Environment Variables:
        GRPO_PHASE_LOG_LEVEL: Log level (0=disabled, 1=main phases, 2=detailed)
        GRPO_PHASE_LOG_FILE: Custom log file path (optional)
        GRPO_PHASE_PRINT_ENABLED: Enable printing to stdout (0=disabled, 1=enabled, default: 1)

    Args:
        output_dir: Directory for log file output
        log_level: Override log level (default: from env or 1)
        log_file: Override log file path (default: from env or {output_dir}/grpo_phase.log)
        print_enabled: Override print to stdout setting (default: from env or True)
    """

    # Phase names mapping
    PHASE_NAMES = {
        1: "ROLLOUT",
        2: "REWARD_COMPUTATION",
        3: "ADVANTAGE_COMPUTATION",
        4: "TEXT_EMBEDDING_PRECOMPUTATION",
        5: "MICRO_BATCH_GRADIENT_ACCUMULATION",
        6: "OPTIMIZATION",
        7: "LOGGING_AND_CLEANUP",
    }

    # Phases to print to stdout (simplified view)
    # These map to the conceptual RL stages: Rollout + Policy Update
    PRINT_PHASES = {
        1: "Rollout",           # ROLLOUT -> Rollout
        2: "Reward",            # REWARD_COMPUTATION -> Reward (total)
        5: "Policy Update",     # MICRO_BATCH_GRADIENT_ACCUMULATION -> Policy Update
    }

    def __init__(
        self,
        output_dir: str,
        log_level: Optional[int] = None,
        log_file: Optional[str] = None,
        print_enabled: Optional[bool] = None,
    ):
        """Initialize the PhaseLogger."""
        self._is_rank_zero = self._check_rank_zero()
        self._log_level = self._get_log_level(log_level)
        self._log_file_path = self._get_log_file_path(output_dir, log_file)
        self._print_enabled = self._get_print_enabled(print_enabled)
        self._file_handle = None
        self._phase_start_times: Dict[int, float] = {}
        self._step_start_time: Optional[float] = None
        # Track per-step phase durations for printing summary at step end
        self._current_step_phase_durations: Dict[str, float] = {}

        # Initialize log file for rank 0
        if self._is_rank_zero and self._log_level > 0:
            self._init_log_file()

    def _check_rank_zero(self) -> bool:
        """Check if current process is global rank 0."""
        try:
            if dist.is_initialized():
                return dist.get_rank() == 0
            else:
                # Not distributed, treat as rank 0
                return True
        except Exception:
            # Fallback to single GPU mode
            return True

    def _get_log_level(self, override: Optional[int]) -> int:
        """Get log level from override or environment variable."""
        if override is not None:
            return override
        try:
            return int(os.environ.get("GRPO_PHASE_LOG_LEVEL", "1"))
        except ValueError:
            return 1

    def _get_log_file_path(self, output_dir: str, override: Optional[str]) -> str:
        """Get log file path from override or environment variable."""
        if override is not None:
            return override
        env_path = os.environ.get("GRPO_PHASE_LOG_FILE")
        if env_path:
            return env_path
        return os.path.join(output_dir, "grpo_phase.log")

    def _get_print_enabled(self, override: Optional[bool]) -> bool:
        """Get print enabled setting from override or environment variable."""
        if override is not None:
            return override
        try:
            return os.environ.get("GRPO_PHASE_PRINT_ENABLED", "1") == "1"
        except ValueError:
            return True

    def _init_log_file(self) -> None:
        """Initialize the log file for writing."""
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(self._log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Open file in append mode
            self._file_handle = open(self._log_file_path, "a", encoding="utf-8")

            # Write header
            self._write_line("=" * 80)
            self._write_line(f"GRPO Phase Logger initialized at {self._get_datetime()}")
            self._write_line(f"Log level: {self._log_level}")
            self._write_line(f"Log file: {self._log_file_path}")
            self._write_line("=" * 80)
            self._flush()
        except Exception as e:
            print(f"[WARNING] Failed to initialize phase log file: {e}")
            self._file_handle = None

    def _get_datetime(self) -> str:
        """Get current datetime in YYYY-MM-DD HH:MM:SS format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_line(self, line: str) -> None:
        """Write a line to the log file."""
        if self._file_handle is not None:
            try:
                self._file_handle.write(line + "\n")
            except Exception:
                pass  # Silently ignore write errors

    def _flush(self) -> None:
        """Flush the log file buffer."""
        if self._file_handle is not None:
            try:
                self._file_handle.flush()
            except Exception:
                pass

    def _format_extra_info(self, extra_info: Dict[str, Any]) -> str:
        """Format extra info dictionary as key=value pairs."""
        if not extra_info:
            return ""
        parts = [f"{k}={v}" for k, v in extra_info.items()]
        return " | " + ", ".join(parts)

    def phase_start(
        self,
        step: int,
        phase_num: int,
        phase_name: Optional[str] = None,
        **extra_info: Any,
    ) -> None:
        """
        Log the start of a phase.

        Args:
            step: Current training step number
            phase_num: Phase number (1-7)
            phase_name: Optional custom phase name (default: from PHASE_NAMES)
            **extra_info: Additional key-value pairs to log
        """
        if not self._is_rank_zero or self._log_level < 1:
            return

        # Record start time
        self._phase_start_times[phase_num] = time.perf_counter()

        # Get phase name
        name = phase_name or self.PHASE_NAMES.get(phase_num, f"PHASE_{phase_num}")

        # Format log line
        datetime_str = self._get_datetime()
        extra_str = self._format_extra_info(extra_info)
        line = f"[{datetime_str}][Step {step}] ===== Phase {phase_num}: {name} START ====={extra_str}"

        self._write_line(line)
        self._flush()

    def phase_end(
        self,
        step: int,
        phase_num: int,
        phase_name: Optional[str] = None,
        duration_ms: Optional[float] = None,
        **extra_info: Any,
    ) -> None:
        """
        Log the end of a phase.

        Prints timing to stdout immediately on global rank 0.

        Args:
            step: Current training step number
            phase_num: Phase number (1-7)
            phase_name: Optional custom phase name (default: from PHASE_NAMES)
            duration_ms: Optional duration in milliseconds (auto-calculated if not provided)
            **extra_info: Additional key-value pairs to log
        """
        if not self._is_rank_zero or self._log_level < 1:
            return

        # Calculate duration
        if duration_ms is None and phase_num in self._phase_start_times:
            start_time = self._phase_start_times.pop(phase_num)
            duration_ms = (time.perf_counter() - start_time) * 1000

        # Get phase name
        name = phase_name or self.PHASE_NAMES.get(phase_num, f"PHASE_{phase_num}")

        # Track phase duration for step summary
        if duration_ms is not None:
            self._current_step_phase_durations[name] = duration_ms

        # Format log line
        datetime_str = self._get_datetime()
        duration_str = f" ({duration_ms:.1f}ms)" if duration_ms is not None else ""
        extra_str = self._format_extra_info(extra_info)
        line = f"[{datetime_str}][Step {step}] ===== Phase {phase_num}: {name} END{duration_str} ====={extra_str}"

        self._write_line(line)
        self._flush()

        # Print phase timing to stdout only for key phases (simplified view)
        if self._print_enabled and duration_ms is not None:
            if phase_num in self.PRINT_PHASES:
                print_name = self.PRINT_PHASES[phase_num]
                formatted_duration = self._format_duration(duration_ms)
                print(f"[Step {step}] {print_name}: {formatted_duration}", flush=True)

    def reward_timing(
        self,
        step: int,
        reward_name: str,
        duration_ms: float,
    ) -> None:
        """
        Log timing for individual reward function computation.

        This is printed to stdout and written to log file.

        Args:
            step: Current training step number
            reward_name: Name of the reward function
            duration_ms: Duration in milliseconds
        """
        if not self._is_rank_zero or self._log_level < 1:
            return

        # Write to log file (detailed, under REWARD_COMPUTATION phase)
        datetime_str = self._get_datetime()
        line = f"[{datetime_str}][Step {step}]     ┌─ {reward_name}: {duration_ms:.1f}ms"
        self._write_line(line)
        self._flush()

        # Print to stdout (indented to show it's under Reward phase)
        if self._print_enabled:
            formatted_duration = self._format_duration(duration_ms)
            print(f"[Step {step}]   ┌─ {reward_name}: {formatted_duration}", flush=True)

    def sub_phase(
        self,
        step: int,
        description: str,
        duration_ms: Optional[float] = None,
        **extra_info: Any,
    ) -> None:
        """
        Log a sub-phase or intermediate step.

        Args:
            step: Current training step number
            description: Description of the sub-phase
            duration_ms: Optional duration in milliseconds
            **extra_info: Additional key-value pairs to log
        """
        if not self._is_rank_zero or self._log_level < 2:
            return

        # Format log line
        datetime_str = self._get_datetime()
        duration_str = f" ({duration_ms:.1f}ms)" if duration_ms is not None else ""
        extra_str = self._format_extra_info(extra_info)
        line = f"[{datetime_str}][Step {step}]   > Sub-phase: {description}{duration_str}{extra_str}"

        self._write_line(line)
        self._flush()

    def log_stats(
        self,
        step: int,
        **stats: Any,
    ) -> None:
        """
        Log statistics or metrics.

        Args:
            step: Current training step number
            **stats: Key-value pairs of statistics to log
        """
        if not self._is_rank_zero or self._log_level < 1:
            return

        if not stats:
            return

        # Format log line
        datetime_str = self._get_datetime()
        stats_str = ", ".join(f"{k}: {v}" for k, v in stats.items())
        line = f"[{datetime_str}][Step {step}]   | {stats_str}"

        self._write_line(line)
        self._flush()

    def step_start(self, step: int) -> None:
        """
        Mark the start of a training step.

        Args:
            step: Current training step number
        """
        if not self._is_rank_zero or self._log_level < 1:
            return

        self._step_start_time = time.perf_counter()
        # Reset phase durations for the new step
        self._current_step_phase_durations = {}

        datetime_str = self._get_datetime()
        line = f"[{datetime_str}][Step {step}] ########## TRAINING STEP {step} START ##########"

        self._write_line(line)
        self._flush()

    def step_end(self, step: int, **summary: Any) -> None:
        """
        Mark the end of a training step.

        Prints a timing summary to stdout on global rank 0 with phase durations.

        Args:
            step: Current training step number
            **summary: Summary statistics for the step
        """
        if not self._is_rank_zero or self._log_level < 1:
            return

        # Calculate total step duration
        duration_ms = None
        if self._step_start_time is not None:
            duration_ms = (time.perf_counter() - self._step_start_time) * 1000
            self._step_start_time = None

        datetime_str = self._get_datetime()
        duration_str = f" (Total: {duration_ms:.1f}ms)" if duration_ms is not None else ""

        if summary:
            summary_str = " | " + ", ".join(f"{k}={v}" for k, v in summary.items())
        else:
            summary_str = ""

        line = f"[{datetime_str}][Step {step}] ########## TRAINING STEP {step} END{duration_str}{summary_str} ##########"

        self._write_line(line)
        self._write_line("")  # Empty line for readability
        self._flush()

        # Print total step timing to stdout if enabled
        if self._print_enabled and duration_ms is not None:
            formatted_duration = self._format_duration(duration_ms)
            print(f"[Step {step}] Total: {formatted_duration}", flush=True)

    def _get_short_phase_name(self, phase_name: str) -> str:
        """
        Get a shorter version of the phase name for printing.

        Args:
            phase_name: Full phase name

        Returns:
            Shortened phase name
        """
        short_names = {
            "ROLLOUT": "Rollout",
            "REWARD_COMPUTATION": "Reward",
            "ADVANTAGE_COMPUTATION": "Advantage",
            "TEXT_EMBEDDING_PRECOMPUTATION": "TextEmbed",
            "MICRO_BATCH_GRADIENT_ACCUMULATION": "GradAccum",
            "OPTIMIZATION": "Optim",
            "LOGGING_AND_CLEANUP": "Cleanup",
        }
        return short_names.get(phase_name, phase_name)

    def _format_duration(self, duration_ms: float) -> str:
        """
        Format duration with smart unit selection (h/m/s/ms).

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Formatted duration string with appropriate unit
        """
        if duration_ms < 1000:
            # Less than 1 second: use ms
            return f"{duration_ms:.1f}ms"
        elif duration_ms < 60 * 1000:
            # Less than 1 minute: use seconds
            seconds = duration_ms / 1000
            return f"{seconds:.2f}s"
        elif duration_ms < 60 * 60 * 1000:
            # Less than 1 hour: use minutes and seconds
            total_seconds = duration_ms / 1000
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            return f"{minutes}m{seconds:.1f}s"
        else:
            # 1 hour or more: use hours, minutes and seconds
            total_seconds = duration_ms / 1000
            hours = int(total_seconds // 3600)
            remaining = total_seconds % 3600
            minutes = int(remaining // 60)
            seconds = remaining % 60
            return f"{hours}h{minutes}m{seconds:.0f}s"

    def close(self) -> None:
        """Close the log file and release resources."""
        if self._file_handle is not None:
            try:
                self._write_line("=" * 80)
                self._write_line(f"GRPO Phase Logger closed at {self._get_datetime()}")
                self._write_line("=" * 80)
                self._flush()
                self._file_handle.close()
            except Exception:
                pass
            finally:
                self._file_handle = None

    def __del__(self):
        """Destructor to ensure log file is closed."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Singleton instance for global access
_global_phase_logger: Optional[PhaseLogger] = None


def get_phase_logger() -> Optional[PhaseLogger]:
    """Get the global PhaseLogger instance."""
    return _global_phase_logger


def set_phase_logger(logger: PhaseLogger) -> None:
    """Set the global PhaseLogger instance."""
    global _global_phase_logger
    _global_phase_logger = logger
