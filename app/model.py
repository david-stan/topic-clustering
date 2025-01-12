from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embeddings(sentences):
    """
    Generate sentence embeddings using the all-mpnet-base-v2 model.
    """
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Process sentences in batches to avoid CUDA out of memory error
    batch_size = 8
    sentence_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]

        # Tokenize sentences
        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

        sentence_embeddings.append(batch_embeddings.cpu())

    sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

    return sentence_embeddings