"""
Path utilities for handling Python import paths across different environments.

This module provides functions to set up the Python import path correctly,
which is especially useful in Databricks environments where __file__ may
not work as expected or when running notebooks.
"""
import os
import sys
from typing import List, Optional


def setup_import_paths(
    base_dir: Optional[str] = None,
    additional_paths: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """
    Set up Python import paths for consistent imports across environments.
    
    This function ensures that the project modules can be imported regardless
    of the current working directory. It adds the base directory and its parent
    directories to sys.path, along with any additional specified paths.
    
    Args:
        base_dir: The base directory to use (defaults to current working directory)
        additional_paths: List of additional paths to add
        verbose: Whether to print debugging information
        
    Example:
        >>> from utils.path_utils import setup_import_paths
        >>> setup_import_paths(verbose=True)
        # Now imports will work from any directory within the project
    """
    # Use current working directory if no base dir is provided
    if base_dir is None:
        base_dir = os.getcwd()
    
    if verbose:
        print(f"Base directory: {base_dir}")
    
    # Add base_dir itself
    if base_dir not in sys.path:
        sys.path.append(base_dir)
        if verbose:
            print(f"Added to path: {base_dir}")
    
    # Add parent directory
    parent_dir = os.path.dirname(base_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        if verbose:
            print(f"Added to path: {parent_dir}")
    
    # Add common subdirectories relative to parent
    common_dirs = ["utils", "models", "data"]
    for dir_name in common_dirs:
        dir_path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(dir_path) and dir_path not in sys.path:
            sys.path.append(dir_path)
            if verbose:
                print(f"Added to path: {dir_path}")
    
    # Add grandparent directory
    grandparent_dir = os.path.dirname(parent_dir)
    if grandparent_dir not in sys.path:
        sys.path.append(grandparent_dir)
        if verbose:
            print(f"Added to path: {grandparent_dir}")
    
    # Add any additional paths
    if additional_paths:
        for path in additional_paths:
            if path not in sys.path:
                sys.path.append(path)
                if verbose:
                    print(f"Added to path: {path}")
    
    if verbose:
        print(f"Python path now includes: {', '.join(sys.path)}")