from sklearn.neural_network import MLPClassifier

def create_model(hidden_layer_sizes=(128, 64,), activation='relu',
                 solver='adam', alpha=0.0001, learning_rate='adaptive',
                 learning_rate_init=0.001, random_state=None) -> MLPClassifier:
    """
    Initialize an MLPClassifier with given hyperparameters.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=50,
        random_state=random_state,
        early_stopping=True  # to prevent long training if no improvement
    )

# MLP_PARAM_GRID = {
#     'hidden_layer_sizes': [(128, 64), (64, 64), (128, 128), (256, 128)],
#     'activation': ['relu'],
#     'alpha': [0.0001, 0.001, 0.01],
#     'learning_rate': ['adaptive'],
#     'solver': ['adam'],
#     'max_iter': [(50, 100, 200, 500, 1000],
# }

MLP_PARAM_GRID = {
    'activation': ['relu'],
    'alpha': [0.001],
    'hidden_layer_sizes': [(64, 64)],
    'learning_rate': ['adaptive'],
    'max_iter': [100],
    'solver': ['adam']
}


# INFO:train:Best parameters: {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (64, 64), 'learning_rate': 'adaptive', 'max_iter': 200, 'solver': 'adam'}
# INFO:train:Best recall score (CV): 0.9767