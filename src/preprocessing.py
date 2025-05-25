import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    oversample: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the credit card dataset:
    - Drop non-informative columns (e.g. 'Time').
    - Scale 'Amount'.
    - Split into training and test sets (stratified).
    - Optionally apply SMOTE oversampling on the training set.

    Returns:
    X_train, X_test, y_train, y_test numpy arrays.
    """
    logger.info("Starting preprocessing...")

    # Separate features and target
    X = df.drop(columns=['Class'], errors='ignore')
    y = df['Class'].values

    # Scale the 'Amount' column
    if 'Amount' in X.columns:
        scaler = StandardScaler()
        X['Amount'] = scaler.fit_transform(X[['Amount']])
        logger.info("Scaled 'Amount' feature.")

    # Split data stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Train/test split: {len(y_train)} train samples, {len(y_test)} test samples")

    # Handle imbalance with SMOTE
    if oversample:
        logger.info("Applying SMOTE to balance the training set...")
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE, training class distribution: {np.bincount(y_train)}")

    return X_train, X_test, y_train, y_test
