import pandas as pd
from transformers import BartTokenizer
from sklearn.model_selection import train_test_split
import torch

# Load the BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Add padding token to the tokenizer
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Function to tokenize data
def tokenize_data(texts, summaries, max_input_length=512, max_output_length=150):
    tokenized_inputs = tokenizer(
        texts,
        max_length=max_input_length,
        padding='max_length',  # Padding to max length
        truncation=True,
        return_tensors="pt"  # Changed to PyTorch tensor format
    )
    tokenized_labels = tokenizer(
        summaries,
        max_length=max_output_length,
        padding='max_length',  # Padding to max length
        truncation=True,
        return_tensors="pt"  # Changed to PyTorch tensor format
    )
    return tokenized_inputs, tokenized_labels

# Function to preprocess and split data
def preprocess_data():
    # Load the data from CSV files
    train_data = pd.read_csv("data/train_data.csv")
    test_data = pd.read_csv("data/test_data.csv")
    
    # Extract texts and summaries
    train_texts = train_data['article'].tolist()
    train_summaries = train_data['highlights'].tolist()
    
    test_texts = test_data['article'].tolist()
    test_summaries = test_data['highlights'].tolist()
    
    # Split train data into training and validation sets
    train_texts, val_texts, train_summaries, val_summaries = train_test_split(
        train_texts, train_summaries, test_size=0.1, random_state=42
    )
    
    # Tokenize data
    train_inputs, train_labels = tokenize_data(train_texts, train_summaries)
    val_inputs, val_labels = tokenize_data(val_texts, val_summaries)
    test_inputs, test_labels = tokenize_data(test_texts, test_summaries)
    
    # Save the tokenized data
    torch.save(train_inputs, 'data/train_inputs.pt')
    torch.save(train_labels, 'data/train_labels.pt')
    torch.save(val_inputs, 'data/val_inputs.pt')
    torch.save(val_labels, 'data/val_labels.pt')
    torch.save(test_inputs, 'data/test_inputs.pt')
    torch.save(test_labels, 'data/test_labels.pt')
    
    return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels

if __name__ == "__main__":
    train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = preprocess_data()

    print("Data preprocessing and tokenization completed.")
