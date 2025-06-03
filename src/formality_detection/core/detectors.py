"""Formality detection implementations using various approaches."""

import re
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests
import spacy
import textstat
from loguru import logger
from transformers import pipeline

from ..config import get_settings


class FormalityDetector(ABC):
    """Abstract base class for formality detection methods."""
    
    @abstractmethod
    def detect_formality(self, text: str) -> float:
        """
        Detect formality level of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Float between 0 (informal) and 1 (formal)
        """
        pass
    
    def validate_text(self, text: str) -> str:
        """Validate and clean input text."""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
            
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty after stripping")
            
        settings = get_settings()
        if len(text) > settings.max_text_length:
            logger.warning(f"Text length {len(text)} exceeds maximum {settings.max_text_length}")
            text = text[:settings.max_text_length]
            
        if len(text) < settings.min_text_length:
            logger.warning(f"Text length {len(text)} below minimum {settings.min_text_length}")
            
        return text


class RuleBasedDetector(FormalityDetector):
    """Rule-based formality detection using linguistic heuristics."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize rule-based detector.
        
        Args:
            model_name: Spacy model name to use (defaults to settings)
        """
        settings = get_settings()
        self.model_name = model_name or settings.spacy_model
        
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError as e:
            logger.error(f"Failed to load spaCy model {self.model_name}: {e}")
            raise
            
    def detect_formality(self, text: str) -> float:
        """
        Detect formality using rule-based and linguistic heuristics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Formality score between 0 (informal) and 1 (formal)
        """
        text = self.validate_text(text)
        
        try:
            formal_indicators = 0.0
            
            # 1. Readability scores (higher scores indicate more complex/formal language)
            flesch_kincaid = max(textstat.flesch_kincaid_grade(text), 0)
            formal_indicators += 1 - max(0, 1 - flesch_kincaid / 10)
            
            # 2. Parse text with spaCy
            doc = self.nlp(text)
            
            # Check for interjections and particles (informal indicators)
            has_particles = any(token.pos_ == 'INTJ' for token in doc)
            has_interjections = any(token.pos_ == 'PART' for token in doc)
            formal_indicators += 1 - (0.5 * has_particles + 0.5 * has_interjections)
            
            # 3. Personal pronouns (informal indicator)
            personal_pronouns = {
                "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
                "ourselves", "you", "your", "yours", "yourself", "yourselves"
            }
            
            pronoun_count = sum(1 for token in doc if token.text.lower() in personal_pronouns)
            pronoun_ratio = pronoun_count / len(doc) if len(doc) > 0 else 0
            formal_indicators += (1 - pronoun_ratio)
            
            formality_score = formal_indicators / 3
            
            # Check for contractions (strong informal indicator)
            contraction_pattern = r'\b\w+\'(s|t|re|ve|ll|d|m)\b'
            has_contraction = bool(re.findall(contraction_pattern, text.lower()))
            if has_contraction:
                formality_score = max(0, formality_score - 0.3)  # Penalize but don't zero out
                
            return max(0, min(1, formality_score))  # Ensure within [0, 1]
            
        except Exception as e:
            logger.error(f"Error in rule-based detection: {e}")
            return 0.5  # Return neutral score on error


class HuggingFaceFormalityDetector(FormalityDetector):
    """Formality detection using pre-trained Hugging Face models."""
    
    def __init__(self, model_name: str, reverse_score: bool = False):
        """
        Initialize Hugging Face detector.
        
        Args:
            model_name: Name of the Hugging Face model
            reverse_score: Whether to reverse the score (1 - score)
        """
        self.model_name = model_name
        self.reverse_score = reverse_score
        
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                return_all_scores=True
            )
            logger.info(f"Loaded Hugging Face model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model {model_name}: {e}")
            raise
            
    def detect_formality(self, text: str) -> float:
        """
        Detect formality using a pre-trained Hugging Face model.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Formality score between 0 (informal) and 1 (formal)
        """
        text = self.validate_text(text)
        
        try:
            results = self.classifier(text)
            
            # Find formal class score
            for label_score in results[0]:
                label = label_score['label'].lower()
                if label in ('label_1', 'formal'):
                    score = float(label_score['score'])
                    if self.reverse_score:
                        score = 1 - score
                    return max(0, min(1, score))
                    
            logger.warning(f"No formal label found in model output: {results}")
            return 0.5
            
        except Exception as e:
            logger.error(f"Error in Hugging Face detection: {e}")
            return 0.5


class OpenAIFormalityDetector(FormalityDetector):
    """Formality detection using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI detector.
        
        Args:
            api_key: OpenAI API key (defaults to settings/environment)
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.api_url = settings.openai_api_url
        self.model = settings.openai_model
        self.rate_limit_delay = settings.rate_limit_delay
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        logger.info(f"Initialized OpenAI detector with model: {self.model}")
        
    def detect_formality(self, text: str) -> float:
        """
        Detect formality using OpenAI API with binary response.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Formality score: 0 (informal), 1 (formal), or 0.5 (uncertain)
        """
        text = self.validate_text(text)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        prompt = f"""
        Analyze the formality level of the following text.
        
        Consider aspects like vocabulary, grammatical constructions, contractions, slang, and tone.
        Respond with ONLY one word: either "formal" or "informal".
        
        Text: {text}
        """
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 10
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip().lower()
            
            if "informal" in response_text:
                return 0.0
            elif "formal" in response_text:
                return 1.0
            else:
                logger.warning(f"Unexpected OpenAI response: {response_text}")
                return 0.5
                
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            return 0.5
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            return 0.5
        finally:
            # Rate limiting
            time.sleep(self.rate_limit_delay) 