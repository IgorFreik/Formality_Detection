"""Main entry point for formality detection evaluation."""

import json
from pathlib import Path
from typing import Dict

from loguru import logger

from .config import get_settings
from .core import (
    FormalityEvaluator,
    HuggingFaceFormalityDetector,
    OpenAIFormalityDetector,
    RuleBasedDetector,
)


def create_detector_configs() -> Dict:
    """Create detector configurations based on settings."""
    settings = get_settings()
    
    configs = {
        "rule-based": {
            "detector": RuleBasedDetector(),
            "description": "Rule-based detector using linguistic heuristics"
        }
    }
    
    # Add Hugging Face detectors
    for name, config in settings.hf_models.items():
        try:
            configs[name] = {
                "detector": HuggingFaceFormalityDetector(
                    model_name=config["model_name"],
                    reverse_score=config.get("reverse_score", False)
                ),
                "description": f"Hugging Face model: {config['model_name']}"
            }
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
    
    # Add OpenAI detector if API key is available
    if settings.openai_api_key:
        try:
            configs["openai_gpt4o_mini"] = {
                "detector": OpenAIFormalityDetector(),
                "description": f"OpenAI model: {settings.openai_model}"
            }
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI detector: {e}")
    else:
        logger.warning("OpenAI API key not found, skipping OpenAI detector")
    
    return configs


def main() -> None:
    """Main evaluation function."""
    settings = get_settings()
    
    # Setup paths
    data_path = Path(settings.data_dir) / "dataset.csv"
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        logger.info("Please ensure your dataset is available or run data preparation first")
        return
    
    # Create output directory
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = FormalityEvaluator(output_dir=str(results_dir))
    
    # Create detector configurations
    detector_configs = create_detector_configs()
    
    if not detector_configs:
        logger.error("No detectors available for evaluation")
        return
    
    logger.info(f"Starting evaluation with {len(detector_configs)} detectors")
    logger.info(f"Dataset: {data_path}")
    logger.info(f"Max samples: {settings.max_samples}")
    logger.info(f"Results directory: {results_dir}")
    
    # Evaluate each detector
    all_metrics = {}
    
    for name, config in detector_configs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {name}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'='*50}")
        
        try:
            results_df, metrics = evaluator.evaluate_detector(
                detector=config["detector"],
                data_path=str(data_path),
                detector_name=name,
                max_samples=settings.max_samples,
                save_results=True
            )
            
            all_metrics[name] = metrics
            
            # Save individual metrics
            metrics_path = results_dir / f"metrics_{name}.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
            continue
    
    # Create summary
    if all_metrics:
        logger.info(f"\n{'='*50}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*50}")
        
        # Sort by F1 score
        sorted_results = sorted(
            all_metrics.items(), 
            key=lambda x: x[1].get('f1', 0), 
            reverse=True
        )
        
        for name, metrics in sorted_results:
            logger.info(f"{name:25} | F1: {metrics.get('f1', 0):.4f} | "
                       f"Acc: {metrics.get('accuracy', 0):.4f} | "
                       f"AUC: {metrics.get('roc_auc', 0):.4f}")
        
        # Save summary
        summary_path = results_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"\nSaved evaluation summary to {summary_path}")
        
    else:
        logger.error("No successful evaluations completed")


if __name__ == "__main__":
    main()
