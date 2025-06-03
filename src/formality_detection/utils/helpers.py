import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict containing the JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path where to save the file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise


def load_dataset(file_path: Path, max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        max_samples: Maximum number of samples to load
        
    Returns:
        DataFrame containing the dataset
    """
    try:
        df = pd.read_csv(file_path)
        if max_samples is not None:
            df = df.head(max_samples)
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset from {file_path}: {e}")
        raise


def calculate_metrics(predictions: List[float], true_labels: List[float]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for formality detection.
    
    Args:
        predictions: List of predicted formality scores
        true_labels: List of true formality scores
        
    Returns:
        Dictionary containing the metrics
    """
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'mse': mean_squared_error(true_labels, predictions),
            'rmse': mean_squared_error(true_labels, predictions, squared=False),
            'mae': mean_absolute_error(true_labels, predictions),
            'r2': r2_score(true_labels, predictions)
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        raise 