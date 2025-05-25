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
        max_iter=300,
        random_state=random_state,
        early_stopping=True  # to prevent long training if no improvement
    )

MLP_PARAM_GRID = {
    'hidden_layer_sizes': [(128, 64), (64, 64), (128,), (256, 128)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['adaptive'],
    'solver': ['adam', 'sgd'],
}