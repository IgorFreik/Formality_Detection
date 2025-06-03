from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import spacy
import textstat
import re
from transformers import pipeline
import requests
from loguru import logger


class BaseFormalityDetector(ABC):
    """Base class for all formality detectors."""
    
    @abstractmethod
    def detect_formality(self, text: str) -> float:
        """
        Detect the formality level of the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            float: A score between 0 (informal) and 1 (formal)
        """
        pass


class RuleBasedDetector(BaseFormalityDetector):
    """Rule-based formality detector using linguistic heuristics."""
    
    def __init__(self) -> None:
        """Initialize the rule-based detector."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("Failed to load spaCy model. Please run: python -m spacy download en_core_web_sm")
            raise

    def detect_formality(self, text: str) -> float:
        """
        Detect formality using rule-based and linguistic heuristics.
        
        Args:
            text: The text to analyze
            
        Returns:
            float: A score between 0 (informal) and 1 (formal)
        """
        if not text.strip():
            return 0.5
            
        formal_indicators = 0.0

        # 1. Readability scores
        try:
            flesch_kincaid = max(textstat.flesch_kincaid_grade(text), 0)
            formal_indicators += 1 - max(0, 1 - flesch_kincaid / 10)
        except Exception as e:
            logger.warning(f"Error calculating readability score: {e}")

        # 2. Linguistic features
        try:
            doc = self.nlp(text)
            
            # Check for interjections and particles
            has_particles = any(token.pos_ == 'INTJ' for token in doc)
            has_interjections = any(token.pos_ == 'PART' for token in doc)
            formal_indicators += 1 - (0.5 * has_particles + 0.5 * has_interjections)

            # Check for personal pronouns
            personal_pronouns = {
                "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
                "ourselves", "you", "your", "yours", "yourself", "yourselves"
            }
            pronoun_count = sum(1 for token in doc if token.text.lower() in personal_pronouns)
            pronoun_ratio = pronoun_count / len(doc)
            formal_indicators += (1 - pronoun_ratio)
        except Exception as e:
            logger.warning(f"Error analyzing linguistic features: {e}")

        formality_score = formal_indicators / 3

        # Check for contractions
        try:
            has_contraction = bool(re.findall(r'\b\w+\'(s|t|re|ve|ll|d|m)\b', text.lower()))
            if has_contraction:
                formality_score = 0
        except Exception as e:
            logger.warning(f"Error checking for contractions: {e}")

        return max(0.0, min(1.0, formality_score))


class HuggingFaceFormalityDetector(BaseFormalityDetector):
    """Formality detector using Hugging Face models."""
    
    def __init__(self, model_name: str, reverse_score: bool = False) -> None:
        """
        Initialize the Hugging Face formality detector.
        
        Args:
            model_name: Name of the Hugging Face model to use
            reverse_score: Whether to reverse the formality score
        """
        self.reverse_score = reverse_score
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                return_all_scores=True
            )
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model {model_name}: {e}")
            raise

    def detect_formality(self, text: str) -> float:
        """
        Detect formality using a pre-trained Hugging Face model.
        
        Args:
            text: The text to analyze
            
        Returns:
            float: A score between 0 (informal) and 1 (formal)
        """
        if not text.strip():
            return 0.5
            
        try:
            results = self.classifier(text)
            
            for label_score in results[0]:
                if label_score['label'] in ('LABEL_1', 'formal'):
                    score = label_score['score']
                    return 1 - score if self.reverse_score else score
                    
            return 0.5
        except Exception as e:
            logger.error(f"Error in formality detection: {e}")
            return 0.5


class OpenAIFormalityDetector(BaseFormalityDetector):
    """Formality detector using OpenAI API."""
    
    def __init__(self, api_key: str) -> None:
        """
        Initialize the OpenAI formality detector.
        
        Args:
            api_key: OpenAI API key
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def detect_formality(self, text: str) -> float:
        """
        Detect formality using OpenAI API.
        
        Args:
            text: The text to analyze
            
        Returns:
            float: A score between 0 (informal) and 1 (formal)
        """
        if not text.strip():
            return 0.5
            
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
            "model": "gpt-4o-mini-2024-07-18",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 10
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip().lower()

            if "informal" in response_text:
                return 0.0
            elif "formal" in response_text:
                return 1.0
            else:
                logger.warning(f"Unexpected API response: {response_text}")
                return 0.5

        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return 0.5 