.PHONY: help install install-dev test lint format clean build docs

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	pip install -e .

install-dev:  ## Install package in development mode with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	pytest tests/ -v --cov=src/formality_detection --cov-report=html --cov-report=term-missing

lint:  ## Run linting checks
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Format code
	black src/ tests/
	isort src/ tests/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

spacy-model:  ## Download required spaCy model
	python -m spacy download en_core_web_sm

setup: install-dev spacy-model  ## Complete development setup

evaluate-all:  ## Evaluate all detectors on dataset
	formality-detect evaluate data/dataset.csv

evaluate-rule:  ## Evaluate rule-based detector only
	formality-detect evaluate data/dataset.csv --detector rule-based --max-samples 1000

predict-example:  ## Run prediction example
	formality-detect predict "Hello, how are you doing today?" 