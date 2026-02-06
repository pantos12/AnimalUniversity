from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repository root based on this file's location."""
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return the data directory, defaulting to ./data."""
    env = os.getenv("AU_DATA_DIR")
    return Path(env).expanduser().resolve() if env else get_repo_root() / "data"


def get_models_dir() -> Path:
    """Return the models directory, defaulting to ./models."""
    env = os.getenv("AU_MODELS_DIR")
    return Path(env).expanduser().resolve() if env else get_repo_root() / "models"


def get_runs_dir() -> Path:
    """Return the runs directory, defaulting to ./data/runs."""
    env = os.getenv("AU_RUNS_DIR")
    return Path(env).expanduser().resolve() if env else get_data_dir() / "runs"
