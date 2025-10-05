from ModelDefinitionandTokenizer import SimpleTokenizer
from ModelDefinitionandTokenizer import SimpleLLM
import torch

# Define our vocabulary and tokenizer
# <pad> is for padding sequences to the same length
# <unk> is for unknown words
VOCAB = {'<pad>': 0, '<unk>': 1, 'dog': 2, 'cat': 3, 'tigger': 4, 'chases': 5, 'the': 6}
tokenizer = SimpleTokenizer(VOCAB)
VOCAB_SIZE = len(VOCAB)
EMBED_DIM = 16 # An arbitrary dimension for the embedding space

# Instantiate the model
model_to_save = SimpleLLM(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)

# --- This is the standard and recommended way to save a model ---
# It only saves the learned parameters (weights and biases).
torch.save(model_to_save.state_dict(), 'simple_llm_state_elon.pth')

print("Model's state_dict has been saved to 'simple_llm_state_elon.pth'")