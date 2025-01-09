from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from app.config import DATA_PATH

nltk.download('stopwords')
stop = set(stopwords.words('english'))

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def load_data():
    """Load dataset from CSV."""
    df = pd.read_csv(DATA_PATH)

    # Create 'cleaned_text' column by copying the 'full_content' column
    df_preprocess = df.copy()
    df_preprocess = df_preprocess[df_preprocess['source_id'] != 'the-times-of-india']
    df_preprocess = df_preprocess[~df_preprocess.full_content.isin(['nan'])]
    df_preprocess.drop_duplicates(['full_content'], inplace=True)
    df_preprocess = df_preprocess.sample(frac=0.2, random_state=42)
    df_preprocess['cleaned_text'] = df_preprocess['full_content']

    # Apply the cleaning function
    df_processed = cleaning(df_preprocess, 'cleaned_text')

    return df_processed.cleaned_text.tolist()

def remove_emojis(x):
    regex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regex_pattern.sub(r'', str(x))

def clean_text(text):
    """Clean text data."""
    text = remove_emojis(text)
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = text.lower()  # Convert to lowercase
    return text

def remove_numbers(text):
    """Remove numbers from text."""
    return re.sub(r'\d+', '', text)

def unify_whitespace(text):
    """Replace multiple whitespaces with a single whitespace."""
    return re.sub(r'\s+', ' ', text)

def remove_symbols(text):
    """Remove symbols from text."""
    return re.sub(r'[^a-zA-Z0-9?!.,]+', ' ', text)

def remove_punctuation(text):
    """Remove punctuation from text."""
    return "".join([char for char in text if char not in ('!', ',', '.', '?', ';', ':', '-', "'")])

def remove_stopwords(text):
    """Remove stopwords from text."""
    return " ".join([word.lower() for word in text.split() if word.lower() not in stop])

def cleaning(df, column):
    """Apply cleaning operations to a DataFrame column."""
    df[column] = df[column].apply(clean_text)
    df[column] = df[column].apply(remove_numbers)
    df[column] = df[column].apply(unify_whitespace)
    df[column] = df[column].apply(remove_symbols)
    df[column] = df[column].apply(remove_punctuation)
    df[column] = df[column].apply(remove_stopwords)
    return df

# Dataset for managing text inputs
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {key: val.squeeze(0) for key, val in encoded.items()}

# def generate_embeddings(texts):
#     """Generate BERT embeddings for a list of texts."""
#     embeddings = []

#     model.eval()

#     for text in texts:
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
#     return embeddings

# Generate embeddings using GPU
def generate_embeddings(texts, batch_size=16, max_length=512):
    """
    Generate BERT embeddings for a list of texts using GPU acceleration.

    Args:
        texts (list): List of text strings.
        batch_size (int): Number of texts processed in a batch.
        max_length (int): Maximum token length for BERT input.

    Returns:
        torch.Tensor: Tensor of embeddings.
    """
    # Prepare dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch in dataloader:
            # Move batch data to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Generate embeddings
            outputs = model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

            # Pooling: Mean of all tokens in the sequence
            pooled_embeddings = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)
            embeddings.append(pooled_embeddings.cpu())  # Move to CPU to save memory

    # Concatenate all embeddings
    return torch.cat(embeddings, dim=0)
