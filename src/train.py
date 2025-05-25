import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score
from joblib import dump

from data_loader import load_data
from preprocessing import preprocess_data
from model import create_model, MLP_PARAM_GRID

logger = logging.getLogger(__name__)

def train_model(data_path: str, model_output_path: str):
    """
    Train the MLP model with hyperparameter tuning.
    Saves the best model to disk.
    """
    # Load and preprocess data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, test_size=0.2, random_state=42, oversample=True)

    # Define model and GridSearchCV
    mlp = create_model(random_state=42)
    scoring = make_scorer(recall_score, pos_label=1)

    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=MLP_PARAM_GRID,
        scoring=scoring,
        n_jobs=-1,
        cv=3,
        verbose=2
    )

    logger.info("Starting grid search for hyperparameters (optimizing recall)...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best recall score (CV): {grid_search.best_score_:.4f}")

    # Save the trained model for later evaluation
    dump(best_model, model_output_path)
    logger.info(f"Model saved to {model_output_path}")
