from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # Model configurations
    max_samples: int = Field(10_000, description="Maximum number of samples to process")
    
    # File paths
    data_dir: Path = Field(Path("data"), description="Directory containing data files")
    results_dir: Path = Field(Path("results"), description="Directory for storing results")
    artifacts_dir: Path = Field(Path("artifacts"), description="Directory for storing artifacts")
    
    # Model configurations
    detector_configs: list[dict] = [
        {
            "name": "rule-based",
            "type": "rule_based",
        },
        {
            "name": "deberta-large-formality-ranker",
            "type": "huggingface",
            "model_name": "s-nlp/deberta-large-formality-ranker",
            "reverse_score": True,
        },
        {
            "name": "roberta-base-formality-ranker",
            "type": "huggingface",
            "model_name": "s-nlp/roberta-base-formality-ranker",
        },
        {
            "name": "xlmr_formality_classifier",
            "type": "huggingface",
            "model_name": "s-nlp/xlmr_formality_classifier",
        },
        {
            "name": "openai_gpt4o_mini",
            "type": "openai",
        },
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings() 