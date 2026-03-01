"""
Data Loader Module

This module contains classes and utilities for loading and processing
various types of datasets using a unified, standardized format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from .data_loader_basic import DataLoaderBase
from .data_loader import DataLoaderKC
from .data_loader_sqlite import DataLoaderSQLite
# from .data_loader_perfile import DataLoaderPerFile

__all__ = [
    "create_data_loader",
    "DataLoaderBase",
    "DataLoaderSQLite",
    "DataLoaderKC",  # TODO: Remove Later or Edit to Unified Loader
    # "DataLoaderPerFile",
]


# Loader registry mapping loader names to classes
LOADER_REGISTRY = {
    "sqlite": DataLoaderSQLite,
    # "perfile": DataLoaderPerFile,
    "standard": DataLoaderSQLite,  # Unified loader (default)
    "unified": DataLoaderSQLite,  # Alias for standard TODO: create later
}


def create_data_loader(
    data_path: str | Path | None = None,
    data_root: str | Path | None = None,
    task_db_name: str | None = None,
    loader_type: str = "standard",
    loader_config: Dict[str, Any] | None = None,
) -> DataLoaderBase:
    """Factory function to create a data loader.
    
    Args:
        data_path: Path to the data .db file path (e.g., "bike_1/default_task.db")
                   If provided, you don't need to provide `data_root` or `task_db_name`.
        data_root: Root directory of the dataset (containing .db files).
                   Use this together with `task_db_name`.
        task_db_name: Specific name of the task database.
                      If not provided when using `data_root`, finds the first non-gt_data .db file.
        loader_type: Type of loader to create. Options:
            - "standard" or "unified": DataLoaderSQLite (default) - unified loader with standard format
        loader_config: Additional configuration dict for the loader (optional)
            - "data_main": Base data directory (default: "dataset/")
            - "strict": If True, only accept standard format (default: False)
            - "config": Alternative way to pass config (same as loader_config)
    
    Returns:
        DataLoaderBase: Instance of the Data Loader
        
    Raises:
        ValueError: If loader_type is not recognized or if both data_path and data_root are provided
        
    Examples:
        >>> # Using data_path with direct .db file
        >>> loader = create_data_loader(
        ...     data_path="bike_1/default_task.db",
        ...     loader_config={"data_main": "dataset/"}
        ... )
        
        >>> # Using data_root and task_db_name
        >>> loader = create_data_loader(
        ...     data_root="dataset/spider_sqlite/college_2",
        ...     task_db_name="college_2.db",
        ...     loader_config={"data_main": "dataset/"}
        ... )
        
        >>> # Direct usage (recommended)
        >>> from core.data_loader import DataLoaderSQLite
        >>> loader = DataLoaderSQLite(data_path="bike_1/default_task", config={"data_main": "dataset/"})
    """
    loader_type = loader_type.lower()
    
    if loader_type not in LOADER_REGISTRY:
        available = ", ".join(LOADER_REGISTRY.keys())
        logging.error(f"[create_data_loader] Unknown loader type: '{loader_type}'. "
                     f"Available loaders: {available}")
        raise ValueError(
            f"Unknown loader type: '{loader_type}'. "
            f"Available loaders: {available}"
        )
    
    loader_class = LOADER_REGISTRY[loader_type]
    loader_config = loader_config or {}
    
    # Extract config from loader_config if provided
    config = loader_config.pop("config", loader_config)
    
    # Pass parameters to loader class
    # The loader class will handle validation of mutually exclusive parameters
    return loader_class(
        data_path=data_path,
        data_root=data_root,
        task_db_name=task_db_name,
        config=config,
        **loader_config
    )
