import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import string
import re
from  fruitDataset import FruitDataset
from simpleTransformer import SimpleTransformer

# Read and clean text
with open("fruits.txt", "r") as f:
    text = f.read()

# Or use inline:
# text = data.strip()

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s\n]', '', text)  # Remove punctuation
    return text

cleaned = clean_text(text)
chars = sorted(list(set(cleaned)))
vocab_size = len(chars)
print("Vocabulary size:", vocab_size)
print("Chars:", ''.join(chars))

# Map characters to indices and back
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode text
encoded = np.array([char_to_idx[ch] for ch in cleaned])

#==========================================================
# Create dataset
seq_len = 25
dataset = FruitDataset(encoded, seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#==========================================================
# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(vocab_size=vocab_size, seq_len=seq_len).to(device)
#============Step 7: Train the Model===========================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
epochs = 100

from tqdm import tqdm

for epoch in range(epochs):
    total_loss = 0
    for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

 #===================================================================
 #============================save model=============================
# Save model weights
torch.save(model.state_dict(), "fruit_llm_model.pth")
#Step 8: Generate New Text(Inference)
def generate_text(model, seed_text, length=100, temperature=1.0):
    model.eval()
    input_text = clean_text(seed_text.lower())
    input_ids = [char_to_idx[ch] for ch in input_text[-seq_len:]]

    while len(input_ids) < seq_len:
        input_ids.insert(0, 0)  # pad if needed

    input_ids = input_ids[-seq_len:]  # truncate
    generated = input_ids.copy()

    for _ in range(length):
        x = torch.tensor([generated[-seq_len:]], device=device)
        with torch.no_grad():
            logits = model(x)
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
        generated.append(next_id)

    return ''.join([idx_to_char[i] for i in generated])

# Try generating
print(generate_text(model, "Apple", length=150, temperature=0.8))
print(generate_text(model, "Strawberry", length=150, temperature=0.8))