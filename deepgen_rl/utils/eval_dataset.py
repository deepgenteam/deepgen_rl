# Copyright 2025 Ruihang Li and DeepGen Team @ Shanghai Innovation Institute

"""
Evaluation Dataset Module for DeepGen-RL.

This module provides classes for loading and managing evaluation datasets:
- EvalDatasetConfig: Parses YAML configuration files for multiple evaluation datasets
- EvalPromptDataset: Loads prompts from various file formats (.txt, .jsonl, .parquet, .csv)

Supports scoring configuration for post-generation evaluation:
- unigenbench: UniGenBench evaluation using VLM judge model

CSV Format (for UniGenBench):
    index,prompt,sub_dims
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import yaml
from torch.utils.data import Dataset
from datasets import load_dataset

# Import scoring config parser
from deepgen_rl.evaluation.unigenbench import (
    UniGenBenchScoringConfig,
    parse_scoring_config,
)


@dataclass
class EvalDatasetInfo:
    """Information about a single evaluation dataset."""
    name: str
    path: str
    num_samples: int = 0
    duplicates: int = 1  # Number of images to generate per prompt
    cfg_scale: Optional[float] = None  # Per-dataset CFG scale (None = use default)
    num_inference_steps: Optional[int] = None  # Per-dataset inference steps (None = use default)
    # Scoring configuration (e.g., unigenbench)
    scoring_config: Optional[UniGenBenchScoringConfig] = None


class EvalPromptDataset(Dataset):
    """
    Dataset for loading evaluation prompts from various file formats.

    Supported formats:
    - .txt: One prompt per line
    - .jsonl: JSON Lines with 'prompt' or 'caption' field
    - .parquet: Parquet file with 'prompt' or 'caption' column
    - .csv: CSV file with 'index', 'prompt', 'sub_dims' columns

    Each sample returns a dict with:
    - prompt: The text prompt
    - index: Global index within this dataset
    - dataset_name: Name of the dataset this sample belongs to
    - duplicate_idx: Index of duplicate (0 to duplicates-1) when duplicates > 1
    """

    def __init__(
        self,
        file_path: str,
        dataset_name: str,
        duplicates: int = 1,
        cfg_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        scoring_config: Optional[UniGenBenchScoringConfig] = None,
    ):
        """
        Initialize the evaluation prompt dataset.

        Args:
            file_path: Path to the prompts file
            dataset_name: Name identifier for this dataset
            duplicates: Number of images to generate per prompt (default: 1)
            cfg_scale: CFG scale for this dataset (None = use global default)
            num_inference_steps: Number of inference steps for this dataset (None = use global default)
            scoring_config: Scoring configuration (e.g., UniGenBench)
        """
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.duplicates = max(1, duplicates)  # Ensure at least 1
        self.cfg_scale = cfg_scale
        self.num_inference_steps = num_inference_steps
        self.scoring_config = scoring_config
        self.prompts: List[Dict[str, Any]] = []

        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Evaluation dataset file not found: {file_path}")

        # Load prompts based on file extension
        if file_path.endswith(".txt"):
            self._load_txt(file_path)
        elif file_path.endswith(".jsonl"):
            self._load_jsonl(file_path)
        elif file_path.endswith(".parquet"):
            self._load_parquet(file_path)
        elif file_path.endswith(".csv"):
            self._load_csv(file_path)
        else:
            raise ValueError(
                f"Unsupported file format for evaluation dataset: {file_path}. "
                f"Supported formats: .txt, .jsonl, .parquet, .csv"
            )

    def _load_txt(self, file_path: str) -> None:
        """Load prompts from a text file (one prompt per line)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                prompt = line.strip()
                if prompt:  # Skip empty lines
                    # Create duplicate entries for each prompt
                    for dup_idx in range(self.duplicates):
                        self.prompts.append({
                            "prompt": prompt,
                            "index": idx,
                            "dataset_name": self.dataset_name,
                            "duplicate_idx": dup_idx,
                        })

    def _load_jsonl(self, file_path: str) -> None:
        """Load prompts from a JSON Lines file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    prompt = item.get('prompt', item.get('caption', ''))
                    if prompt:
                        # Create duplicate entries for each prompt
                        for dup_idx in range(self.duplicates):
                            self.prompts.append({
                                "prompt": prompt,
                                "index": idx,
                                "dataset_name": self.dataset_name,
                                "metadata": item,  # Keep original data for reference
                                "duplicate_idx": dup_idx,
                            })
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {idx} in {file_path}: {e}")

    def _load_parquet(self, file_path: str) -> None:
        """Load prompts from a Parquet file."""
        dataset = load_dataset("parquet", data_files={"data": file_path})["data"]
        for idx, item in enumerate(dataset):
            prompt = item.get("prompt", item.get("caption", ""))
            if prompt:
                # Create duplicate entries for each prompt
                for dup_idx in range(self.duplicates):
                    self.prompts.append({
                        "prompt": prompt,
                        "index": idx,
                        "dataset_name": self.dataset_name,
                        "duplicate_idx": dup_idx,
                    })

    def _load_csv(self, file_path: str) -> None:
        """
        Load prompts from a CSV file.

        Expected columns: index, prompt
        (sub_dims column is used by UniGenBenchScorer for evaluation, not needed here)
        """
        df = pd.read_csv(file_path)

        # Check for required columns
        if "prompt" not in df.columns:
            raise ValueError(
                f"CSV file {file_path} missing required 'prompt' column. "
                f"Found columns: {list(df.columns)}"
            )

        for row_idx, row in df.iterrows():
            # Use 'index' column if available, otherwise use row index
            if "index" in df.columns:
                idx = int(row["index"])
            else:
                idx = row_idx

            prompt = row["prompt"]
            if not prompt or pd.isna(prompt):
                continue

            # Create duplicate entries for each prompt
            for dup_idx in range(self.duplicates):
                self.prompts.append({
                    "prompt": prompt,
                    "index": idx,
                    "dataset_name": self.dataset_name,
                    "duplicate_idx": dup_idx,
                })

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.prompts[idx]

    def get_prompts_for_wandb(self, num_images: int) -> List[Dict[str, Any]]:
        """
        Get a fixed set of prompts for wandb visualization.

        Returns the first `num_images` prompts. These will be used consistently
        across all evaluation steps for comparison.

        Args:
            num_images: Number of prompts to return

        Returns:
            List of prompt dictionaries
        """
        return self.prompts[:min(num_images, len(self.prompts))]


class EvalDatasetConfig:
    """
    Configuration for multiple evaluation datasets.

    Parses YAML config files with the following format:
    ```yaml
    datasets:
      - name: geneval
        path: /path/to/geneval_prompts.jsonl
      - name: unigenbench_en
        path: unigenbench/test_prompts_en.csv
        duplicates: 4
        scoring: unigenbench
    ```

    The path can be absolute or relative to the config file directory.
    """

    def __init__(self, config_path: str):
        """
        Initialize from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config format is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Evaluation dataset config not found: {config_path}")

        self.config_path = config_path
        self.config_dir = os.path.dirname(os.path.abspath(config_path))

        # Load and parse YAML config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Validate config structure
        if not isinstance(self.config, dict):
            raise ValueError(f"Invalid config format in {config_path}: expected dict, got {type(self.config)}")

        if 'datasets' not in self.config:
            raise ValueError(f"Missing 'datasets' key in config file: {config_path}")

        if not isinstance(self.config['datasets'], list):
            raise ValueError(f"'datasets' must be a list in config file: {config_path}")

        # Parse dataset configurations
        self.datasets_info: List[EvalDatasetInfo] = []
        self._parse_datasets()

    def _parse_datasets(self) -> None:
        """Parse and validate each dataset configuration."""
        for idx, ds_cfg in enumerate(self.config['datasets']):
            if not isinstance(ds_cfg, dict):
                raise ValueError(f"Dataset entry {idx} must be a dict, got {type(ds_cfg)}")

            # Get required fields
            name = ds_cfg.get('name')
            path = ds_cfg.get('path')

            if not name:
                raise ValueError(f"Dataset entry {idx} missing required 'name' field")
            if not path:
                raise ValueError(f"Dataset entry {idx} missing required 'path' field")

            # Resolve relative paths
            if not os.path.isabs(path):
                path = os.path.join(self.config_dir, path)

            # Verify file exists
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Dataset file not found: {path} "
                    f"(specified in '{name}' entry of {self.config_path})"
                )

            # Get optional duplicates field (default: 1)
            duplicates = ds_cfg.get('duplicates', 1)
            if not isinstance(duplicates, int) or duplicates < 1:
                print(f"Warning: Invalid duplicates value {duplicates} for dataset '{name}', using 1")
                duplicates = 1

            # Get optional per-dataset inference parameters
            cfg_scale = ds_cfg.get('cfg_scale', None)
            num_inference_steps = ds_cfg.get('num_inference_steps', None)

            # Validate cfg_scale if provided
            if cfg_scale is not None:
                try:
                    cfg_scale = float(cfg_scale)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid cfg_scale value {cfg_scale} for dataset '{name}', using default")
                    cfg_scale = None

            # Validate num_inference_steps if provided
            if num_inference_steps is not None:
                try:
                    num_inference_steps = int(num_inference_steps)
                    if num_inference_steps < 1:
                        raise ValueError("Must be positive")
                except (ValueError, TypeError):
                    print(f"Warning: Invalid num_inference_steps value {num_inference_steps} for dataset '{name}', using default")
                    num_inference_steps = None

            # Parse scoring configuration (supports string or dict format)
            scoring_value = ds_cfg.get('scoring', None)
            scoring_config = parse_scoring_config(scoring_value, self.config_dir)

            self.datasets_info.append(EvalDatasetInfo(
                name=name,
                path=path,
                duplicates=duplicates,
                cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                scoring_config=scoring_config,
            ))

    def create_datasets(self) -> List[EvalPromptDataset]:
        """
        Create EvalPromptDataset instances for all configured datasets.

        Returns:
            List of EvalPromptDataset instances
        """
        datasets = []
        for ds_info in self.datasets_info:
            dataset = EvalPromptDataset(
                file_path=ds_info.path,
                dataset_name=ds_info.name,
                duplicates=ds_info.duplicates,
                cfg_scale=ds_info.cfg_scale,
                num_inference_steps=ds_info.num_inference_steps,
                scoring_config=ds_info.scoring_config,
            )
            ds_info.num_samples = len(dataset)
            datasets.append(dataset)

            # Build info string for logging
            dup_info = f" (x{ds_info.duplicates} duplicates)" if ds_info.duplicates > 1 else ""
            cfg_info = f", cfg_scale={ds_info.cfg_scale}" if ds_info.cfg_scale is not None else ""
            steps_info = f", steps={ds_info.num_inference_steps}" if ds_info.num_inference_steps is not None else ""
            if ds_info.scoring_config:
                lang_info = f"/{ds_info.scoring_config.language}" if hasattr(ds_info.scoring_config, 'language') else ""
                scoring_info = f", scoring={ds_info.scoring_config.type}{lang_info}"
            else:
                scoring_info = ""
            print(f"Loaded eval dataset '{ds_info.name}': {len(dataset)} samples{dup_info}{cfg_info}{steps_info}{scoring_info} from {ds_info.path}")

        return datasets

    def get_dataset_names(self) -> List[str]:
        """Get names of all configured datasets."""
        return [ds.name for ds in self.datasets_info]

    def __repr__(self) -> str:
        datasets_str = ", ".join([f"'{ds.name}'" for ds in self.datasets_info])
        return f"EvalDatasetConfig(config_path='{self.config_path}', datasets=[{datasets_str}])"
