"""Formality Detection Package.

A comprehensive toolkit for detecting formality in text using various approaches including
rule-based methods, transformer models, and LLM APIs.
"""

from .config import Settings, get_settings
from .core import (
    FormalityDetector,
    FormalityEvaluator,
    HuggingFaceFormalityDetector,
    OpenAIFormalityDetector,
    RuleBasedDetector,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Core classes
    "FormalityDetector",
    "RuleBasedDetector",
    "HuggingFaceFormalityDetector", 
    "OpenAIFormalityDetector",
    "FormalityEvaluator",
    # Configuration
    "Settings",
    "get_settings",
    # Package info
    "__version__",
] 