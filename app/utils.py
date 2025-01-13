import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import RESULTS_PATH
from app.model import assign_main_category

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

import re
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation

# Extract sub-cluster topics
def extract_sub_topics(cluster_texts, n_top_words=5):
    """
    Extract sub-topics for a cluster using TF-IDF.
    """
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Get top words for the cluster
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    sorted_indices = tfidf_scores.argsort()[::-1]
    top_words = [feature_names[i] for i in sorted_indices[:n_top_words]]
    
    return list(dict.fromkeys(top_words))  # Remove duplicates

# Generate hierarchical labels
def generate_hierarchical_labels(clustered_data):
    """
    Generate hierarchical labels for clusters, including main categories and sub-level descriptions.
    """
    hierarchical_labels = {}

    for cluster_id in clustered_data["cluster"].unique():
        # Get all text from the current cluster
        cluster_texts = clustered_data[clustered_data["cluster"] == cluster_id]["text"].values
        combined_text = " ".join(cluster_texts)

        # Assign main category
        main_category = assign_main_category(combined_text)

        # Extract sub-topics
        sub_topics = extract_sub_topics(cluster_texts)
        refined_sub_topics = remove_redundant_terms(sub_topics)

        # Create hierarchical label
        hierarchical_labels[cluster_id] = {
            "main_category": main_category,
            "sub_topics": refined_sub_topics,
        }

    return hierarchical_labels

# Save the HDBSCAN object
def save_hdbscan_model(hdbscan_model, file_path="best_hdbscan_model.pkl"):
    """
    Save the HDBSCAN clustering model to a file.

    Args:
        hdbscan_model: The HDBSCAN object to save.
        file_path: The file path where the model will be saved.
    """
    joblib.dump(hdbscan_model, file_path)
    print(f"HDBSCAN model saved to {file_path}")

# Load the HDBSCAN object
def load_hdbscan_model(file_path="best_hdbscan_model.pkl"):
    """
    Load the HDBSCAN clustering model from a file.

    Args:
        file_path: The file path from which the model will be loaded.

    Returns:
        The loaded HDBSCAN object.
    """
    hdbscan_model = joblib.load(file_path)
    print(f"HDBSCAN model loaded from {file_path}")
    return hdbscan_model

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def remove_redundant_terms(sub_topics):
    """
    Remove redundant terms by lemmatizing and filtering out closely related terms.
    """
    unique_terms = set()
    cleaned_topics = []
    for term in sub_topics:
        lemma = lemmatizer.lemmatize(term.lower())  # Lemmatize the term
        if lemma not in unique_terms:
            unique_terms.add(lemma)
            cleaned_topics.append(term)  # Keep the original term for readability
    return cleaned_topics