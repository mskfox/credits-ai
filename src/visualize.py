import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Legit', 'Predicted Fraud'],
                yticklabels=['Actual Legit', 'Actual Fraud'])
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_roc(y_test, y_proba):
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"AUC-ROC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0,1], [0,1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {auc:.4f})')
    plt.show()

def plot_metrics_vs_threshold(y_true, y_proba):
    thresholds = np.linspace(0, 1, 100)
    recalls = []
    specificities = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        # Handle cases where predictions are all 0 or all 1
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            if np.all(y_pred == 0):
                tn = (y_true == 0).sum()
                fn = (y_true == 1).sum()
            else:
                fp = (y_true == 0).sum()
                tp = (y_true == 1).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recalls.append(recall)
        specificities.append(specificity)

    intersection_threshold = None
    for i in range(1, len(thresholds)):
        if (recalls[i-1] - specificities[i-1]) * (recalls[i] - specificities[i]) <= 0:
            intersection_threshold = thresholds[i]
            intersection_value = (recalls[i] + specificities[i]) / 2
            break

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, recalls, label='Recall (Positive Class)')
    plt.plot(thresholds, specificities, label='Specificity (Negative Class)')

    # Add intersection line if found
    if intersection_threshold is not None:
        plt.axvline(x=intersection_threshold, color='red', linestyle='--',
                   label=f'Intersection (Threshold={intersection_threshold:.2f})')
        plt.scatter(intersection_threshold, intersection_value, color='red')

    plt.xlabel('Classification Threshold')
    plt.ylabel('Metric Value')
    plt.title('Threshold vs. Recall/Specificity')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_curve(loss_curve):
    """
    Plot training loss curve over epochs

    Args:
        loss_curve (list): List of loss values per training epoch
    """
    plt.figure(figsize=(8, 6))
    plt.plot(loss_curve, marker='o', markersize=4)
    plt.title("Model Training Loss Curve")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()