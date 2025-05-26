import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    oversample: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess credit card transaction data for fraud detection modeling.

    Steps:
    1. Separate features from target variable
    2. Scale monetary features
    3. Create stratified train/test splits
    4. Optionally balance classes using SMOTE
    5. Remove train/test duplicates

    Args:
        df: Raw input dataframe with transaction data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        oversample: Whether to apply SMOTE oversampling

    Returns:
        Processed train/test features and labels as numpy arrays
    """
    logger.info("Starting data preprocessing pipeline")

    # Initial data preparation
    features, labels = _separate_features_from_target(df)
    features = _scale_amount_feature(features)

    # Create stratified split
    X_train, X_test, y_train, y_test = _create_stratified_split(
        features, labels, test_size, random_state
    )

    # Handle class imbalance
    if oversample:
        X_train, y_train = _balance_classes_with_smote(
            X_train, y_train, X_test, y_test, random_state
        )

    return X_train, X_test, y_train, y_test

def _separate_features_from_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Isolate features from target variable ('Class')"""
    features = df.drop(columns=['Class'], errors='ignore')
    labels = df['Class'].values
    return features, labels

def _scale_amount_feature(features: pd.DataFrame) -> pd.DataFrame:
    """Standardize the 'Amount' feature using z-score normalization"""
    if 'Amount' in features.columns:
        scaler = StandardScaler()
        features['Amount'] = scaler.fit_transform(features[['Amount']])
        logger.info("Successfully scaled 'Amount' feature")
    return features

def _create_stratified_split(
    features: pd.DataFrame,
    labels: np.ndarray,
    test_size: float,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/test splits maintaining class distribution"""
    X_train, X_test, y_train, y_test = train_test_split(
        features.values,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    logger.info(
        f"Created stratified split: {X_train.shape[0]} training samples, "
        f"{X_test.shape[0]} test samples"
    )
    return X_train, X_test, y_train, y_test

def _balance_classes_with_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling and remove duplicates from training data
    that exist in the test set
    """
    logger.info("Beginning class balancing with SMOTE")

    # Apply SMOTE oversampling
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(
        "Class distribution after SMOTE: "
        f"{np.bincount(y_resampled).tolist()}"
    )

    # Remove overlapping samples between train and test
    X_clean, y_clean = _remove_train_test_duplicates(
        X_resampled, y_resampled, X_test
    )
    logger.info(
        "Final training class distribution: "
        f"{np.bincount(y_clean).tolist()}"
    )
    return X_clean, y_clean

def _remove_train_test_duplicates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove duplicate samples from training data that appear in test set"""
    # Convert to DataFrame for easier duplicate detection
    train_df = pd.DataFrame(X_train)
    test_df = pd.DataFrame(X_test)

    # Find overlapping samples
    duplicate_mask = train_df.apply(tuple, axis=1).isin(test_df.apply(tuple, axis=1))
    duplicate_count = duplicate_mask.sum()

    if duplicate_count > 0:
        logger.info(f"Removing {duplicate_count} test duplicates from training data")
        clean_train = train_df[~duplicate_mask]
        clean_labels = y_train[~duplicate_mask]
        return clean_train.values, clean_labels

    logger.info("No overlapping samples found between train/test sets")
    return X_train, y_train