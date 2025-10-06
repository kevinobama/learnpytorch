import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import string
import re
from tqdm import tqdm
import math

datasetFile = "../dataset/english_word_dataset.txt"

# ===============================
# 1️⃣ Data Loading & Cleaning
# ===============================
with open(datasetFile, "r") as f:
    text = f.read()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


cleaned = clean_text(text)
print(f"Total characters: {len(cleaned)}")
print(f"Sample text: {cleaned[:200]}")

# ===============================
# 2️⃣ Vocabulary with Special Tokens
# ===============================
words = cleaned.split()
print(f"Total words: {len(words)}")
print(f"Unique words: {len(set(words))}")

vocab = sorted(list(set(words)))
vocab_size = len(vocab)

# Add special tokens
special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
vocab = special_tokens + vocab
vocab_size = len(vocab)

word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}


# Handle unknown words
def encode_word(w):
    return word_to_idx.get(w, word_to_idx['<UNK>'])


encoded = np.array([encode_word(w) for w in words])
print(f"Vocabulary size: {vocab_size}")


# ===============================
# 3️⃣ Dataset with Train/Val Split
# ===============================
class WordDataset(Dataset):
    def __init__(self, encoded, seq_len=10):
        self.data = encoded
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


seq_len = 10
batch_size = 32

# Create dataset and split
full_dataset = WordDataset(encoded, seq_len=seq_len)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# ===============================
# 4️⃣ FIXED Transformer with Correct Positional Encoding
# ===============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0)]


class ImprovedTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim=256, num_heads=8, ff_hidden=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=seq_len)

        # Use batch_first=False for standard transformer format
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            batch_first=False  # Standard: (seq_len, batch_size, embed_dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x) * math.sqrt(self.embed_dim)  # (batch_size, seq_len, embed_dim)

        # Transpose for transformer: (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        # Add positional encoding
        x = self.pos_encoding(x)  # (seq_len, batch_size, embed_dim)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # Transformer expects (seq_len, batch_size, embed_dim)
        x = self.transformer(x)  # (seq_len, batch_size, embed_dim)

        # Transpose back: (batch_size, seq_len, embed_dim)
        x = x.permute(1, 0, 2)

        logits = self.fc(x)  # (batch_size, seq_len, vocab_size)
        return logits


# ===============================
# 5️⃣ Training with Validation
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ImprovedTransformer(
    vocab_size=vocab_size,
    seq_len=seq_len,
    embed_dim=128,  # Reduced for stability
    num_heads=4,  # Reduced for stability
    ff_hidden=256,  # Reduced for stability
    num_layers=2,  # Reduced for stability
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)


def calculate_accuracy(logits, targets):
    _, predicted = torch.max(logits, dim=-1)
    correct = (predicted == targets).float().sum()
    total = targets.numel()
    return correct / total

# ===============================
# 6️⃣ Load Best Model for Generation
# ===============================
model.load_state_dict(torch.load("best_word_transformer.pth"))
model.eval()


# ===============================
# 7️⃣ Text Generation
# ===============================
def generate_text(model, prompt, length=20, temperature=0.8, top_k=20):
    model.eval()
    words_input = clean_text(prompt).split()

    # Handle input sequence
    input_ids = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in words_input]
    if len(input_ids) > seq_len:
        input_ids = input_ids[-seq_len:]
    elif len(input_ids) < seq_len:
        input_ids = [word_to_idx['<PAD>']] * (seq_len - len(input_ids)) + input_ids

    generated = input_ids.copy()

    for _ in range(length):
        x = torch.tensor([generated[-seq_len:]], device=device, dtype=torch.long)

        with torch.no_grad():
            logits = model(x)  # (1, seq_len, vocab_size)
            logits = logits[0, -1, :] / max(temperature, 0.1)

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        generated.append(next_id)

    # Convert back to words
    generated_words = []
    for idx in generated:
        word = idx_to_word[idx]
        if word not in special_tokens:
            generated_words.append(word)

    return ' '.join(generated_words)


# ===============================
# 8️⃣ Test Generation
# ===============================
print("\n" + "=" * 50)
print("TESTING GENERATION")
print("=" * 50)

test_prompts = ["apple", "the fruit", "I like", "banana is"]

for prompt in test_prompts:
    generated = generate_text(model, prompt, length=10, temperature=0.7, top_k=20)
    print(f"Prompt: '{prompt}'")
    print(f"Generated: {generated}")
    print("-" * 30)