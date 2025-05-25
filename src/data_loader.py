import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """
    Load the credit card transactions dataset from a CSV file.
    Returns a pandas DataFrame with the raw data.
    """
    logger.info(f"Loading data from {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
