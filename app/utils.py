import json
from app.config import RESULTS_PATH

def save_results(clusters, reduced_data):
    """Save clustering results."""
    data = {
        "clusters": clusters.tolist(),
        "reduced_data": reduced_data.tolist()
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f)

def load_results():
    """Load clustering results."""
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)
