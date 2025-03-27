import spacy
import textstat
import re
from transformers import pipeline
import requests


class RuleBasedDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def detect_formality(self, text):
        """Detect formality using rule-based and linguistic heuristics."""
        formal_indicators = 0

        # 1. Readability scores (higher scores indicate more difficult to read/informal language)
        flesch_kincaid = max(textstat.flesch_kincaid_grade(text), 0)
        formal_indicators += 1 - max(0, 1 - flesch_kincaid / 10)

        # 2. Has interjections or particles (informal text tends to have more interjections and particles)
        doc = self.nlp(text)
        has_particles = any(token.pos_ == 'INTJ' for token in doc)
        has_interjections = any(token.pos_ == 'PART' for token in doc)
        formal_indicators += 1 - (0.5 * has_particles + 0.5 * has_interjections)

        # 3. Personal pronouns (informal text tends to use more personal pronouns)
        personal_pronouns = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
                             "ourselves", "you", "your", "yours", "yourself", "yourselves"]

        pronoun_count = sum(1 for token in doc if token.text.lower() in personal_pronouns)
        pronoun_ratio = pronoun_count / len(doc)
        formal_indicators += (1 - pronoun_ratio)

        formality_score = formal_indicators / 3

        # Check for contractions (informal)
        has_contraction = float(len(re.findall(r'\b\w+\'(s|t|re|ve|ll|d|m)\b', text.lower())) > 0)
        if has_contraction:
            formality_score = 0

        return formality_score


class HuggingFaceFormalityDetector:
    def __init__(self, model_name, reverse_score=False):
        self.reverse_score = reverse_score
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True
        )

    def detect_formality(self, text):
        """Detect formality using a pre-trained Hugging Face model."""
        results = self.classifier(text)

        for label_score in results[0]:
            if (label_score['label'] == 'LABEL_1') or (label_score['label'] == 'formal'):  # Formal class
                score = label_score['score']
                if self.reverse_score:
                    return 1 - score
                return score
        return 0.5


class OpenAIFormalityDetector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def detect_formality(self, text):
        """Detect formality using OpenAI API with binary response."""
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

            if "informal" in response_text.lower():
                return 0
            elif "formal" in response_text.lower():
                return 1
            else:
                return 0.5

        except Exception as e:
            print(f"API request failed: {e}")
            return 0.5
