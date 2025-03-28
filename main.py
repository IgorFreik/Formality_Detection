import os
import json
from formality_detector import (
    OpenAIFormalityDetector,
    HuggingFaceFormalityDetector,
    RuleBasedDetector
)
from evaluate import evaluate_formality_detector


MAX_SAMPLES = 10_000
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DETECTOR_CONFIGS = [
    {
        "name": "rule-based",
        "detector": RuleBasedDetector(),
    },
    {
        "name": "deberta-large-formality-ranker",
        "detector": HuggingFaceFormalityDetector(model_name="s-nlp/deberta-large-formality-ranker", reverse_score=True),
    },
    {
        "name": "roberta-base-formality-ranker",
        "detector": HuggingFaceFormalityDetector(model_name="s-nlp/roberta-base-formality-ranker"),
    },
    {
        "name": "xlmr_formality_classifier",
        "detector": HuggingFaceFormalityDetector(model_name="s-nlp/xlmr_formality_classifier"),
    },
    {
        "name": "openai_gpt4o_mini",
        "detector": OpenAIFormalityDetector(OPENAI_API_KEY),
    },
]


if __name__ == '__main__':
    for config in DETECTOR_CONFIGS:
        detector = config["detector"]
        results_df, metrics = evaluate_formality_detector(
            detector,
            "data/dataset.csv",
            detector_name=config["name"],
            max_samples=MAX_SAMPLES,
        )
        print(metrics)

        with open(f"metrics_{config['name']}.json", "w") as f:
            json.dump(metrics, f)
