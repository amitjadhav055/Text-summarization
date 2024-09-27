import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import evaluate
from torch.utils.data import DataLoader, TensorDataset

# Load the pre-trained model and tokenizer
model = BartForConditionalGeneration.from_pretrained('saved_model').to("cuda")  # Using GPU (cuda)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Load the test data
test_inputs = torch.load('data/test_inputs.pt')  # Preprocessed inputs
test_labels = torch.load('data/test_labels.pt')  # Reference labels

# Prepare DataLoader for batch processing
batch_size = 16  # Increase batch size for faster evaluation
test_dataset = TensorDataset(test_inputs['input_ids'], test_labels['input_ids'])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Load ROUGE evaluation metric
rouge = evaluate.load('rouge')

# Initialize lists to store results
predicted_summaries = []
reference_summaries = []

model.eval()

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, labels = batch
        input_ids = input_ids.to("cuda")  # Move data to GPU
        labels = labels.to("cuda")

        # Generate summaries in batches
        summaries = model.generate(input_ids, num_beams=4, max_length=142, early_stopping=True)

        # Decode generated and reference summaries
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
        decoded_references = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]

        # Append to lists for evaluation
        predicted_summaries.extend(decoded_summaries)
        reference_summaries.extend(decoded_references)

# Evaluate using ROUGE
results = rouge.compute(predictions=predicted_summaries, references=reference_summaries)
print("ROUGE Results:", results)
