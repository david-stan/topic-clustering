from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from app.config import DATA_PATH

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def load_data():
    """Load dataset from CSV."""
    df = pd.read_csv(DATA_PATH)
    return df["text"].tolist()

def generate_embeddings(texts):
    """Generate BERT embeddings for a list of texts."""
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return embeddings
