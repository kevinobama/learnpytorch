import torch
import torch.nn as nn

# A simple, custom tokenizer for our vocabulary
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse_vocab = {i: token for token, i in vocab.items()}

    def encode(self, text):
        # Converts a single word to its token ID
        return self.vocab.get(text, self.vocab['<unk>'])

    def decode(self, token_id):
        # Converts a token ID back to a word
        return self.inverse_vocab.get(token_id, '<unk>')

# Define the exact same model architecture used for training
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
        print(f"Model initialized with vocab size: {vocab_size} and embedding dim: {embed_dim}")

    def forward(self, x):
        # x is expected to be a tensor of token IDs
        embedded = self.embedding(x)
        # For simplicity, we'll just average the embeddings if there are multiple
        # In a real transformer, this would be much more complex!
        pooled = embedded.mean(dim=1) if len(embedded.shape) > 2 else embedded
        logits = self.linear(pooled)
        return logits