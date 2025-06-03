import sys
from pathlib import Path

from loguru import logger

from formality_detection.config.settings import settings
from formality_detection.core.detectors import (
    RuleBasedDetector,
    HuggingFaceFormalityDetector,
    OpenAIFormalityDetector
)
from formality_detection.utils.helpers import (
    load_dataset,
    calculate_metrics,
    save_json
)


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        settings.artifacts_dir / "logs" / "formality_detection_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )


def create_detector(config: dict):
    """
    Create a formality detector based on configuration.
    
    Args:
        config: Detector configuration dictionary
        
    Returns:
        An instance of a formality detector
    """
    detector_type = config["type"]
    
    if detector_type == "rule_based":
        return RuleBasedDetector()
    elif detector_type == "huggingface":
        return HuggingFaceFormalityDetector(
            model_name=config["model_name"],
            reverse_score=config.get("reverse_score", False)
        )
    elif detector_type == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI detector")
        return OpenAIFormalityDetector(settings.openai_api_key)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def main():
    """Main entry point for the formality detection application."""
    # Setup logging
    setup_logging()
    logger.info("Starting formality detection")
    
    # Create necessary directories
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "logs").mkdir(exist_ok=True)
    
    # Load dataset
    dataset_path = settings.data_dir / "dataset.csv"
    df = load_dataset(dataset_path, max_samples=settings.max_samples)
    logger.info(f"Loaded dataset with {len(df)} samples")
    
    # Process each detector
    for config in settings.detector_configs:
        detector_name = config["name"]
        logger.info(f"Processing detector: {detector_name}")
        
        try:
            # Create detector
            detector = create_detector(config)
            
            # Get predictions
            predictions = []
            for text in df["text"]:
                score = detector.detect_formality(text)
                predictions.append(score)
            
            # Calculate metrics
            metrics = calculate_metrics(predictions, df["formality_score"].tolist())
            
            # Save results
            results_file = settings.artifacts_dir / f"metrics_{detector_name}.json"
            save_json(metrics, results_file)
            
            logger.info(f"Completed processing {detector_name}")
            logger.info(f"Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error processing detector {detector_name}: {e}")
            continue
    
    logger.info("Formality detection completed")


if __name__ == "__main__":
    main() 