"""Evaluation functionality for formality detectors."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from ..config import get_settings
from .detectors import FormalityDetector


class FormalityEvaluator:
    """Evaluator for formality detection models."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save results (defaults to settings)
        """
        settings = get_settings()
        self.output_dir = Path(output_dir or settings.results_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.threshold_steps = settings.threshold_steps
        
    def evaluate_detector(
        self,
        detector: FormalityDetector,
        data_path: Union[str, Path],
        detector_name: str,
        max_samples: Optional[int] = None,
        save_results: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Evaluate a formality detector on a dataset.
        
        Args:
            detector: FormalityDetector instance to evaluate
            data_path: Path to CSV dataset with 'text' and 'label' columns
            detector_name: Name identifier for the detector
            max_samples: Maximum number of samples to evaluate (optional)
            save_results: Whether to save detailed results to CSV
            
        Returns:
            Tuple of (results_dataframe, metrics_dict)
        """
        # Load and prepare data
        df = self._load_data(data_path, max_samples)
        logger.info(f"Evaluating {detector_name} on {len(df)} samples")
        
        # Run predictions
        y_true, y_pred_proba, results_df = self._predict_batch(detector, df, detector_name)
        
        # Find optimal threshold
        best_threshold = self._find_optimal_threshold(y_true, y_pred_proba)
        
        # Calculate final predictions and metrics
        y_pred = [1 if prob > best_threshold else 0 for prob in y_pred_proba]
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba, best_threshold)
        
        # Add prediction results to dataframe
        results_df['prediction_binary'] = y_pred
        results_df['prediction_label'] = ['Formal' if p == 1 else 'Informal' for p in y_pred]
        results_df['is_correct'] = (results_df['prediction_binary'] == results_df['label'])
        
        # Log results
        self._log_results(metrics, detector_name)
        self._print_confusion_matrix(y_true, y_pred)
        
        # Save results if requested
        if save_results:
            output_path = self.output_dir / f"{detector_name}_results_{int(time.time())}.csv"
            results_df.to_csv(output_path, index=False)
            logger.info(f"Saved detailed results to {output_path}")
            
        return results_df, metrics
        
    def _load_data(self, data_path: Union[str, Path], max_samples: Optional[int]) -> pd.DataFrame:
        """Load and optionally sample dataset."""
        settings = get_settings()
        
        try:
            df = pd.read_csv(data_path).reset_index(drop=True)
            logger.info(f"Loaded dataset with {len(df)} samples")
            
            # Validate required columns
            required_cols = ['text', 'label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Sample if requested
            if max_samples and len(df) > max_samples:
                df = df.sample(max_samples, random_state=settings.random_seed).reset_index(drop=True)
                logger.info(f"Sampled {max_samples} samples for evaluation")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise
            
    def _predict_batch(
        self, 
        detector: FormalityDetector, 
        df: pd.DataFrame, 
        detector_name: str
    ) -> Tuple[List[int], List[float], pd.DataFrame]:
        """Run batch prediction on dataset."""
        settings = get_settings()
        y_true = []
        y_pred_proba = []
        error_count = 0
        
        # Initialize result columns
        df = df.copy()
        df['prediction_proba'] = None
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {detector_name}"):
            try:
                text = row['text']
                if pd.isna(text) or not isinstance(text, str):
                    logger.warning(f"Invalid text at index {idx}: {text}")
                    prediction = 0.5
                    error_count += 1
                else:
                    prediction = detector.detect_formality(text)
                    
                y_true.append(int(row['label']))
                y_pred_proba.append(float(prediction))
                df.at[idx, 'prediction_proba'] = float(prediction)
                
                # Rate limiting for API-based detectors  
                if 'openai' in detector_name.lower():
                    time.sleep(settings.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                y_true.append(int(row['label']))
                y_pred_proba.append(0.5)
                df.at[idx, 'prediction_proba'] = 0.5
                error_count += 1
                
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during prediction")
            
        return y_true, y_pred_proba, df
        
    def _find_optimal_threshold(self, y_true: List[int], y_pred_proba: List[float]) -> float:
        """Find optimal classification threshold using F1 score."""
        thresholds = np.linspace(0, 1, self.threshold_steps)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = [1 if prob > threshold else 0 for prob in y_pred_proba]
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        return float(best_threshold)
        
    def _calculate_metrics(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        y_pred_proba: List[float],
        threshold: float
    ) -> Dict:
        """Calculate evaluation metrics."""
        try:
            metrics = {
                "optimal_threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, y_pred_proba),
                "processed_samples": len(y_pred),
                "total_samples": len(y_true),
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return default metrics on error
            metrics = {
                "optimal_threshold": threshold,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.5,
                "processed_samples": len(y_pred),
                "total_samples": len(y_true),
            }
            
        return metrics
        
    def _log_results(self, metrics: Dict, detector_name: str) -> None:
        """Log evaluation results."""
        logger.info(f"\nEvaluation Results for {detector_name}:")
        logger.info(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Processed {metrics['processed_samples']} samples")
        
    def _print_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> None:
        """Print confusion matrix."""
        try:
            predictions_df = pd.DataFrame({
                'Actual': ['Formal' if v == 1 else 'Informal' for v in y_true],
                'Predicted': ['Formal' if v == 1 else 'Informal' for v in y_pred]
            })
            confusion = pd.crosstab(
                predictions_df['Actual'],
                predictions_df['Predicted'],
                rownames=['Actual'],
                colnames=['Predicted']
            )
            logger.info("\nConfusion Matrix:")
            logger.info(f"\n{confusion}")
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {e}")
            
    def compare_detectors(
        self,
        detectors: Dict[str, FormalityDetector],
        data_path: Union[str, Path],
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple detectors on the same dataset.
        
        Args:
            detectors: Dict mapping detector names to detector instances
            data_path: Path to evaluation dataset
            max_samples: Maximum samples to evaluate
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_results = []
        
        for name, detector in detectors.items():
            logger.info(f"Evaluating detector: {name}")
            try:
                _, metrics = self.evaluate_detector(
                    detector, data_path, name, max_samples, save_results=False
                )
                metrics['detector_name'] = name
                comparison_results.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            
            # Save comparison results
            output_path = self.output_dir / f"detector_comparison_{int(time.time())}.csv"
            comparison_df.to_csv(output_path, index=False)
            logger.info(f"Saved comparison results to {output_path}")
            
            return comparison_df
        else:
            logger.warning("No successful evaluations to compare")
            return pd.DataFrame() 