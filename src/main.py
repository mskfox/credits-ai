import argparse
import logging
import sys

from train import train_model
from evaluate import evaluate_model
from data_loader import load_data
from preprocessing import preprocess_data
from joblib import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection with MLPClassifier"
    )
    parser.add_argument('--train', action='store_true', help="Train a new model")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate an existing model")
    parser.add_argument('--data', type=str, required=True, help="Path to CSV data file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to save/load model file")
    args = parser.parse_args()

    if args.train:
        logger.info("Running training mode...")
        train_model(data_path=args.data, model_output_path=args.model_path)

    elif args.evaluate:
        logger.info("Running evaluation mode...")
        logger.info("Loading data and model...")
        df = load_data(args.data)
        _, X_test, _, y_test = preprocess_data(df, test_size=0.2, random_state=42, oversample=False)
        model = load(args.model_path)
        evaluate_model(model, X_test, y_test)

    else:
        logger.error("You must specify either --train or --evaluate")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
