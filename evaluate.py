import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import os
from tqdm import tqdm


def evaluate_formality_detector(detector, csv_path, detector_name: str, max_samples=None, output_path=None):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    print(f"Loaded dataset with {len(df)} samples")

    if (max_samples is not None) and (len(df) > max_samples):
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)
        print(f"Reduced dataset to {max_samples} samples for testing")

    y_true = []
    y_pred_proba = []
    error_count = 0

    df['prediction_proba'] = None
    df['prediction_binary'] = None
    df['prediction_label'] = None

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        prediction = detector.detect_formality(text)

        y_true.append(int(row['label']))
        y_pred_proba.append(prediction)
        df.at[idx, 'prediction_proba'] = prediction

        if 'openai' in detector_name.lower():
            time.sleep(0.1)  # Avoid rate limiting

    # Find optimal threshold
    thresholds = np.linspace(0, 1, 101)
    best_acc = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = [1 if prob > threshold else 0 for prob in y_pred_proba]
        acc = f1_score(y_true, y_pred, zero_division=0)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    # Apply optimal threshold to predictions
    y_pred = [1 if prob > best_threshold else 0 for prob in y_pred_proba]

    for idx, (pred_prob, pred) in enumerate(zip(y_pred_proba, y_pred)):
        df.at[idx, 'prediction_binary'] = pred
        df.at[idx, 'prediction_label'] = "Formal" if pred == 1 else "Informal"

    df['is_correct'] = df.apply(
        lambda x: x['prediction_binary'] == x['label']
        if pd.notnull(x['prediction_binary']) else None,
        axis=1
    )

    # Calculate metrics
    metrics = {
        "optimal_threshold": best_threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "processed_samples": len(y_pred),
        "total_samples": len(df),
        "failed_samples": error_count
    }

    print("\nEvaluation Results:")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Successfully processed {metrics['processed_samples']} out of {metrics['total_samples']} samples")

    # Plot a confusion matrix
    predictions = pd.DataFrame({
        'Actual': ['Formal' if v == 1 else 'Informal' for v in y_true],
        'Predicted': ['Formal' if v == 1 else 'Informal' for v in y_pred]
    })
    confusion = pd.crosstab(
        predictions['Actual'],
        predictions['Predicted'],
        rownames=['Actual'],
        colnames=['Predicted']
    )
    print("\nConfusion Matrix:")
    print(confusion)

    # Save raw results
    if not output_path:
        if not os.path.exists('results'):
            os.makedirs('results')
        output_path = f"results/{detector_name}_results_{int(time.time())}.csv"

    df.to_csv(output_path, index=False)
    print(f"Saved raw results to {output_path}")

    return df, metrics
