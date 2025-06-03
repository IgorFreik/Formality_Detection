# Formality Detection Project Documentation

## Scripts Description
<!-- [Github Repository](https://github.com/IgorFreik/Formality_Detection) -->
- `data_preparation.py` — Data preparation.
- `formality_detector.py` — All approaches to formality detection wrapped in Python classes.
- `evaluation.py` — Evaluation of given detectors on a given dataset.
- `main.py` — Script to run evaluation on a prepared dataset using defined detectors.
- `plot_results.py` — Plot and save visualizations and artifacts for final examination.

---

## Dataset Used

The data used in this project was taken from the authors of the following study, where they curated a human-annotated formality scores dataset:

> Ellie Pavlick and Joel Tetreault. 2016. *An empirical analysis of formality in online communication.* Transactions of the Association for Computational Linguistics, 4:61–74.

The dataset is publicly available at:
[Formality Corpus](http://www.seas.upenn.edu/~nlp/resources/formality-corpus.tgz)

Important Note: Although synthetic data (e.g., rephrasing the same sentences using an LLM in formal/informal tone) could enrich the dataset, it was decided to rely solely on human-written data. This ensures that formality detection aligns with real-world applications, where only human-written text needs formality detection.

---

## Tools Used
1. Data Preparation: `pandas`, `imblearn` (for majority undersampling to mitigate class imbalance).
2. First Approach (Rules & Linguistic Heuristics): `spacy`, `regex`, `textstat`.
3. Second Approach (Fine-Tuned Transformers): `transformers` framework.
4. Third Approach (LLM-as-a-Judge): `requests` library (OpenAI API for GPT models).
5. Plotting Results: `seaborn`, `matplotlib`, `sklearn` (e.g., ROC curves), `pandas`.

---

## Metrics Used

Since this is a binary classification problem, standard classification metrics were used:
1. ROC-AUC
2. F1 Score
3. Precision/Recall
4. Accuracy

To convert probabilistic predictions into binary classifications, an optimal cutoff threshold strategy was applied. The final score corresponds to the best value among 100 evenly distributed thresholds in the interval [0,1].

---

## Approach

1. Defining labeling strategy
    - Decided to follow the literature-based approach [(source)](https://arxiv.org/pdf/2204.08975) of using two classes: “formal” and “informal”.
    - Even though this approach might not perfectly align with use-case-specific requirements, it still provides insights into the performance of different formality detection methods.

2. Data preparation steps (Based on an EDA Analysis)
    - Remove overly long/short texts If a long text contains mixed formality, it's unclear what the correct label should be. Texts <5 characters are often unreadable.
    - Remove texts with conflicting scores → Unreliable if annotators disagreed.
    - Remove texts with fewer than 5 annotations → Less reliable.
    - Final evaluation dataset was built from four subdatasets, ensuring semantic class balance by undersampling the majority class.

3. Final dataset statistics
    - Total samples: 3,178 texts

4. Implementing LLM-as-a-Judge Approach
    - Model used: `gpt4o-mini`
    - Expected output: Binary classification (“formal” or “informal”) using few-shot prompting.

---

## Ideas

1. Task-Specific Considerations:
    - If evaluating sentence-level formality, additional features might be extracted.
    - Cases where the LLM failed often involved lack of capitalization are ADDTEXT
---

## Results Summary

1. Performance Comparison:
    - Fine-tuned LLM significantly outperformed both rule-based and LLM-as-a-judge approaches.
    - Rule-based approach was the fastest (~10x faster than LLM-based methods).
    - However, batch processing & knowledge distillation can significantly reduce fine-tuned model latency.

---

## Discussion & Future Work

1. Hybrid model approach
    - A combination of LLM-based and rule-based approaches appears ideal based on misclassified samples.
    - Current LLM-as-a-Judge method was zero-shot; adding a few good examples might improve performance.

2. Potential improvements in data preparation
    - For texts with a formality score around 0, manual review could help refine the dataset.
    - The dataset should be filtered to better reflect the business-specific definition of formality.

3. False negatives (FN) vs. false positives (FP)
    - Errors are equally distributed between FN & FP.
    - Most incorrect predictions occur when the sentence is neutral, yet the true label is formal/informal.
---

## Challenges Encountered

1. Subjectivity of Formality
    - Formality perception varies among individuals, making labeling inherently difficult.

2. Challenges in Rule-Based Approaches
    - Several ideas failed to improve predictions in the first approach, including:
        - Using sentiment levels (assumed high sentiment = informal, but it didn’t correlate well).
        - Measuring lexical diversity (word repetition & rarity didn’t predict formality effectively).
        - Splitting sentences & measuring average word length (no significant impact).

---

<!-- ## How to run and reproduce results

For instructions on running the project and reproducing the results, please refer to the GitHub repository’s README: -->


<!-- [GitHub Repository](https://github.com/IgorFreik/Formality_Detection) -->



# How to run

1. Install the required libraries using the following command:
```bash
# Set up a virutal environment if you want
# python3 -m venv venv
# source venv/bin/activate
pip install -r requirements.txt
```

2. Run the following command to prepare the data:
```bash
python prepare_data.py
```

3. Run the following command to evaluate the models:
```bash
python main.py
```

4. Get the ROC/AUC curves; confusion matrixes; samples of incorrectly classified texts using the following command:
```bash
python plot_results.py
```

