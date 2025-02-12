"""File handling utilities for the application."""

import os
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Union, List
import logging
import time

logger = logging.getLogger(__name__)

def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists.
        
    Returns:
        Path object of the ensured directory.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def file_hash(file_path: Union[str, Path]) -> str:
    """Generate MD5 hash for file content validation.
    
    Args:
        file_path: Path to the file to hash.
        
    Returns:
        MD5 hash of the file content.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def safe_delete(path: Union[str, Path]) -> bool:
    """Safely delete a file or directory.
    
    Args:
        path: Path to delete.
        
    Returns:
        True if deletion was successful, False otherwise.
    """
    try:
        path = Path(path)
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        return True
    except Exception as e:
        logger.error(f"Error deleting {path}: {str(e)}")
        return False

def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search in.
        pattern: Glob pattern to match files against.
        recursive: Whether to search recursively.
        
    Returns:
        List of matching file paths.
    """
    path = Path(directory)
    if recursive:
        return list(path.rglob(pattern))
    return list(path.glob(pattern))

def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
    """Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Size of the file in bytes, or None if file doesn't exist.
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None

def clean_directory(
    directory: Union[str, Path],
    pattern: str = "*",
    max_age_days: Optional[float] = None
) -> int:
    """Clean files in a directory matching certain criteria.
    
    Args:
        directory: Directory to clean.
        pattern: Glob pattern for files to clean.
        max_age_days: Maximum age of files to keep (None keeps all).
        
    Returns:
        Number of files deleted.
    """
    path = Path(directory)
    deleted_count = 0
    
    if not path.exists():
        return 0
        
    for file_path in path.glob(pattern):
        if not file_path.is_file():
            continue
            
        should_delete = False
        if max_age_days is not None:
            age_days = (
                time.time() - file_path.stat().st_mtime
            ) / (24 * 3600)
            should_delete = age_days > max_age_days
            
        if should_delete and safe_delete(file_path):
            deleted_count += 1
            
    return deleted_count 