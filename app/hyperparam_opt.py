from hyperopt import fmin, tpe, Trials, space_eval, hp, STATUS_OK
from functools import partial

from app.clustering import generate_umap, generate_clusters, score_clusters

def objective_wrapper(embeddings, n_neighbors, n_components, min_cluster_size, min_samples, random_state=None):
    """Objective function for hyperparameter optimization.
    
    Args:
        params (dict): Dictionary of hyperparameters.
        embeddings (np.ndarray): Array of message embeddings.
        label_lower (int): Lower bound for the number of unique cluster labels.
        label_upper (int): Upper bound for the number of unique cluster labels.
    
    """
    umap_embeddings = generate_umap(embeddings, n_components, n_neighbors, random_state)
    clusters = generate_clusters(umap_embeddings, min_cluster_size, min_samples)
    return clusters

def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperparameter optimization.

    Args:
        params (dict): Dictionary of hyperparameters.
        embeddings (np.ndarray): Array of message embeddings.
        label_lower (int): Lower bound for the number of unique cluster labels.
        label_upper (int): Upper bound for the number of unique cluster labels.

    Returns:
        dict: Dictionary containing the loss value, number of unique cluster labels, and status.

    """
    # n_neighbors = trial.suggest_int('n_neighbors', 2, 100)
    # min_dist = trial.suggest_float('min_dist', 0.01, 0.5)
    # min_cluster_size = trial.suggest_int('min_cluster_size', 2, 100)

    clusters = objective_wrapper(
        embeddings,
        n_neighbors=params["n_neighbors"],
        n_components=params["n_components"],
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        random_state=params["random_state"],
    )

    label_count, cost = score_clusters(clusters, prob_threshold=0.05)

    if (label_count < label_lower) or (label_count > label_upper):
        penalty = 1.0
    else:
        penalty = 0.0

    loss = cost + penalty

    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}



def bayesian_optimization(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayesian search on hyperparameter space using hyperopt

    Arguments:
        embeddings: embeddings to use
        space: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', and 'random_state' and
               values that use built-in hyperopt functions to define
               search spaces for each
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters
        max_evals: int, maximum number of parameter combinations to try

    Saves the following to instance variables:
        best_params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'min_samples', and 'random_state' and
               values associated with lowest cost scenario tested
        best_clusters: HDBSCAN object associated with lowest cost scenario
                       tested
        trials: hyperopt trials object for search

    """
    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings, label_lower=label_lower, label_upper=label_upper)
    best = fmin(fmin_objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

    best_params = space_eval(space, best)

    print('best parameters:', best_params)
    print(f"label_count: {trials.best_trial['result']['label_count']}")

    best_clusters = objective_wrapper(embeddings, **best_params)

    return best_params, best_clusters, trials

hspace = {
    'n_neighbors': hp.choice('n_neighbors', range(3, 32)),
    'min_cluster_size': hp.choice('min_cluster_size', range(2, 32)),
    "n_components": hp.choice("n_components", range(2, 32)),
    "min_samples": hp.choice("min_samples", range(2, 32)),
    "random_state": 42
}