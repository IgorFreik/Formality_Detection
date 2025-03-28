import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import pandas as pd
import os


def plot_roc(df, save_path="roc_curve.png"):
    """Plots and saves the ROC curve."""
    y_true = df['label']
    y_scores = df['prediction_proba']

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()

    plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(df, save_path="confusion_matrix.png"):
    """Plots and saves the confusion matrix."""
    y_true = df['label']
    y_pred = df['prediction_binary']

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.savefig(save_path)
    plt.show()


def print_incorrect_predictions(df, save_path='incorrect_predictions.txt'):
    """Prints and saves a few incorrectly predicted samples."""
    incorrect = df[df['is_correct'] == False]
    false_positives = incorrect[incorrect['label'] == 0]
    false_negatives = incorrect[incorrect['label'] == 1]

    with open(save_path, 'w') as f:
        f.write("FALSE POSITIVES:\n")
        f.write("---------------\n")
        for _, row in false_positives[['text', 'label', 'prediction_binary']].head().iterrows():
            f.write(f"Text: {row['text']}\n")
            f.write(f"True Label: {row['label']}\n")
            f.write(f"Predicted Label: {row['prediction_binary']}\n")
            f.write("---\n")

        f.write("\nFALSE NEGATIVES:\n")
        f.write("---------------\n")
        for _, row in false_negatives[['text', 'label', 'prediction_binary']].head().iterrows():
            f.write(f"Text: {row['text']}\n")
            f.write(f"True Label: {row['label']}\n")
            f.write(f"Predicted Label: {row['prediction_binary']}\n")
            f.write("---\n")
    return incorrect[['text', 'label', 'prediction_binary']]


if __name__ == "__main__":
    for results_path in os.listdir("results"):
        df = pd.read_csv(results_path)
        plot_roc(df)
        plot_confusion_matrix(df)
        print_incorrect_predictions(df)
