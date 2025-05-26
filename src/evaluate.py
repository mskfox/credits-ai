import logging
import numpy as np

from sklearn.metrics import classification_report

from visualize import plot_confusion, plot_roc, plot_metrics_vs_threshold

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate the trained model on test data, printing metrics and plotting
    confusion matrix and ROC curve.
    """
    logger.info("Evaluating model on test set...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold = 0.9
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])
    logger.info("Classification Report:\n" + report)

    plot_confusion(y_test, y_pred)
    plot_roc(y_test, y_proba)
    plot_metrics_vs_threshold(y_test, y_proba)