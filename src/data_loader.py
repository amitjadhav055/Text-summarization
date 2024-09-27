import os
import pandas as pd
from datasets import load_dataset

def load_data():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")

    # Create 'data/' directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Load the dataset from Hugging Face
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Convert to Pandas DataFrame
    train_data = pd.DataFrame(dataset['train'])
    test_data = pd.DataFrame(dataset['test'])

    # Save the dataset to CSV files
    train_data.to_csv(os.path.join(data_dir, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(data_dir, "test_data.csv"), index=False)

    print(f"Data saved to CSV files in {data_dir}.")

    return train_data, test_data

if __name__ == "__main__":
    train, test = load_data()
    print(train.head())
