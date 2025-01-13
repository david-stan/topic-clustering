import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

from app.config import DATA_PATH

nltk.download('stopwords')
nltk.download('punkt')
nltk_stopwords = set(stopwords.words("english"))
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemma = WordNetLemmatizer()


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

    text =  df_processed.cleaned_text.tolist()

    all_sentences = []

    for intent in text:
        for sentence in nltk.sent_tokenize(intent):
            if len(sentence.split()) > 4:
                all_sentences.append(sentence)

    return all_sentences

# Save embeddings
def save_embeddings(embeddings, file_path):
    """
    Save embeddings to a NumPy .npy file.

    Args:
        embeddings (torch.Tensor): Tensor of embeddings to save.
        file_path (str): Path to save the file.
    """
    # Convert to NumPy array and save
    np.save(file_path, embeddings.numpy())
    print(f"Embeddings saved to {file_path}")

# Load embeddings
def load_saved_embeddings(file_path):
    """
    Load embeddings from a saved .npy file.

    Args:
        file_path (str): Path to the .npy file.

    Returns:
        numpy.ndarray: Loaded embeddings.
    """
    return np.load(file_path)

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
    # text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
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

# def remove_stopwords(text):
#     """Remove stopwords from text."""
#     return " ".join([word.lower() for word in text.split() if word.lower() not in stop])

def Lemmatize(text):
    """
    Preprocess text by cleaning, tokenizing, removing stopwords, lemmatizing, and filtering verbs.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())  # Remove non-alphabetic characters
    tokens = nltk.word_tokenize(text)  # Tokenize text
    tokens = [lemma.lemmatize(word) for word in tokens if word not in nltk_stopwords]  # Lemmatize and remove stopwords
    
    # Filter out verbs
    pos_tags = nltk.pos_tag(tokens)
    filtered_tokens = [word for word, pos in pos_tags if pos not in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")]
    
    return " ".join(filtered_tokens)

def cleaning(df, column):
    """Apply cleaning operations to a DataFrame column."""
    df[column] = df[column].apply(clean_text)
    df[column] = df[column].apply(remove_numbers)
    df[column] = df[column].apply(unify_whitespace)
    df[column] = df[column].apply(remove_symbols)
    df[column] = df[column].apply(remove_punctuation)
    # df[column] = df[column].apply(remove_stopwords)
    df[column] = df[column].apply(Lemmatize)
    return df
