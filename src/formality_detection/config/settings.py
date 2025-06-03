"""Configuration settings for the formality detection system."""

import os
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini-2024-07-18", env="OPENAI_MODEL")
    openai_api_url: str = Field(
        default="https://api.openai.com/v1/chat/completions", 
        env="OPENAI_API_URL"
    )
    
    # Data Configuration
    data_dir: str = Field(default="data", env="DATA_DIR")
    results_dir: str = Field(default="results", env="RESULTS_DIR")
    artifacts_dir: str = Field(default="artifacts", env="ARTIFACTS_DIR")
    
    # Model Configuration
    spacy_model: str = Field(default="en_core_web_sm", env="SPACY_MODEL")
    max_samples: int = Field(default=10_000, env="MAX_SAMPLES")
    random_seed: int = Field(default=42, env="RANDOM_SEED")
    
    # Processing Configuration
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    rate_limit_delay: float = Field(default=0.1, env="RATE_LIMIT_DELAY")
    max_text_length: int = Field(default=400, env="MAX_TEXT_LENGTH")
    min_text_length: int = Field(default=5, env="MIN_TEXT_LENGTH")
    
    # Evaluation Configuration  
    threshold_steps: int = Field(default=101, env="THRESHOLD_STEPS")
    
    # Hugging Face Models Configuration
    hf_models: Dict[str, Dict] = Field(
        default={
            "deberta-large-formality-ranker": {
                "model_name": "s-nlp/deberta-large-formality-ranker",
                "reverse_score": True,
            },
            "roberta-base-formality-ranker": {
                "model_name": "s-nlp/roberta-base-formality-ranker", 
                "reverse_score": False,
            },
            "xlmr_formality_classifier": {
                "model_name": "s-nlp/xlmr_formality_classifier",
                "reverse_score": False,
            },
        }
    )
    
    @validator("openai_api_key")
    def validate_openai_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate OpenAI API key format if provided."""
        if v and not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
        
    @validator("max_samples")
    def validate_max_samples(cls, v: int) -> int:
        """Ensure max_samples is positive."""
        if v <= 0:
            raise ValueError("max_samples must be positive")
        return v
        
    @validator("batch_size")
    def validate_batch_size(cls, v: int) -> int:
        """Ensure batch_size is positive."""
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings() 