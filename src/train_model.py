import torch
from transformers import BartForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training

# Load preprocessed data
train_inputs = torch.load('data/train_inputs.pt')
train_labels = torch.load('data/train_labels.pt')
val_inputs = torch.load('data/val_inputs.pt')
val_labels = torch.load('data/val_labels.pt')

# Initialize the BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
model.resize_token_embeddings(model.config.vocab_size + 1)  # Resize for the extra [PAD] token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare DataLoader
batch_size = 1  # Keep batch size small to avoid OOM issues
train_dataset = TensorDataset(train_inputs['input_ids'], train_labels['input_ids'])
val_dataset = TensorDataset(val_inputs['input_ids'], val_labels['input_ids'])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Set up gradient scaler for mixed precision
scaler = GradScaler()

# Define training loop
epochs = 5  # Increase epochs slightly
gradient_accumulation_steps = 8  # Accumulate gradients every 8 steps

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0

    for i, batch in enumerate(train_dataloader):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Zero gradients every 'gradient_accumulation_steps'
        if i % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        # Mixed precision training with autocast
        with autocast():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Update weights every 'gradient_accumulation_steps'
        if (i + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        # Print progress
        if i % 10 == 0:
            print(f"Batch {i}/{len(train_dataloader)}, Loss: {loss.item()}")

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

    # Validation step
    model.eval()  # Set the model to evaluation mode
    val_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained('saved_model')
print("Model training completed and saved.")
