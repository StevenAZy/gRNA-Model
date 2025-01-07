import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from model import GuideRNAModel
from dataset import PRDataset

hidden_dim = 512
num_heads = 8
ff_dim = 2048
num_decoder_layers = 3
vocab_size = 100  # Example size, should be larger depending on the RNA alphabet


model = GuideRNAModel(hidden_dim, num_heads, ff_dim, num_decoder_layers, vocab_size)
# Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler with warmup and decay
def lr_schedule(epoch, warmup_steps=4000, max_lr=1e-4):
    if epoch < warmup_steps:
        return max_lr * (epoch / warmup_steps)
    else:
        return max_lr * (warmup_steps / (epoch ** 0.5))

scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

# Loss function (CrossEntropyLoss)
criterion = nn.CrossEntropyLoss()

# Training loop (simplified)
epochs = 40
batch_size = 64

train_dataset = PRDataset(file_path='dataset_test.txt')
train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

for epoch in range(epochs):
    model.train()
    for i, data in enumerate(train_dataloader):
        # Forward pass
        optimizer.zero_grad()
        output = model(data)  # Exclude the last token from tgt for decoder input
        
        # Compute loss (ignore padding)
        loss = criterion(output.view(-1, vocab_size), tgt[1:].view(-1))  # Shift tgt for prediction

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Scheduler step
        scheduler.step()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
