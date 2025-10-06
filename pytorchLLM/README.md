act as AI engineer, I'm learning pyTorch, I'd like to build simple LLM in pyTorch
Of course. Let's build a simple character-level Language Model from scratch using PyTorch. This model will learn to predict the next character in a sequence, which is a fundamental concept behind larger LLMs.

We'll follow these key steps:

1.  **Data Preparation**: Convert your text dataset into numerical tensors that the model can understand.
2.  **Model Architecture**: Define a simple neural network using an Embedding, an LSTM, and a Linear layer.
3.  **Training Loop**: Write the code to train the model on your data.
4.  **Text Generation**: Use the trained model to generate new sentences.

-----

### \#\# 1. Data Preparation & Tokenization 🧠

First, we need to process the text. We'll create a **vocabulary** of all unique words and assign an integer to each. We also need to create input-output pairs. For the sentence "A book is a set", the pairs would be:

  * Input: `[A]` -\> Target: `book`
  * Input: `[A, book]` -\> Target: `is`
  * Input: `[A, book, is]` -\> Target: `a`
  * and so on...

<!-- end list -->

```python
import torch
import torch.nn as nn
import torch.optim as optim
import re

# Your provided dataset
raw_text = """
New York: New York is a city in the US.
Computer: A computer is a tool that helps people work, learn, and create things like documents, games, and websites.
Book: A book is a set of written or printed pages, usually bound together, used for reading.
Teacher: A teacher is a person who helps students learn new things at school.
Water: Water is a clear liquid that people, animals, and plants need to live.
Sun: The sun is a big, bright star in the sky that gives us light and heat.
Happy: Happy means feeling joy or pleasure.
Run: To run means to move quickly on foot.
Family: A family is a group of people related by blood or love, like parents and children.
School: A school is a place where students go to learn from teachers.
Apple: An apple is a round fruit that is usually red, green, or yellow.
Music: Music is sounds made by singing or playing instruments, often for enjoyment.
Dream: A dream is a story your mind creates while you sleep, or a goal you hope to achieve.
Kind: Kind means being friendly, helpful, and caring toward others.
Time: Time is what we measure in seconds, minutes, and hours; it tells us when things happen.
Learn: To learn means to gain new knowledge or skills by studying or practicing.
Home: Home is the place where you live, usually with your family.
Friend: A friend is someone you like and enjoy spending time with.
Food: Food is what people and animals eat to stay alive and healthy.
Nature: Nature includes everything in the world that is not made by humans, like trees, rivers, and animals.
"""

# --- Clean and tokenize the text ---
# Remove definitions (e.g., "New York:"), convert to lowercase, and split into words
sentences = [line.split(':')[1].strip() for line in raw_text.strip().split('\n')]
text = ' '.join(sentences).lower()
words = re.findall(r'\b\w+\b', text)

# --- Create the vocabulary ---
vocab = sorted(list(set(words)))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")

# --- Create input-target pairs ---
data = []
for sentence in sentences:
    sentence_words = re.findall(r'\b\w+\b', sentence.lower())
    tokenized_sentence = [word_to_idx[w] for w in sentence_words]
    for i in range(1, len(tokenized_sentence)):
        input_seq = tokenized_sentence[:i]
        target_word = tokenized_sentence[i]
        data.append((input_seq, target_word))

print(f"Created {len(data)} input-target pairs.")
# Example pair: ([word_to_idx['a'], word_to_idx['book']], word_to_idx['is'])
```

-----

### \#\# 2. Model Architecture ⚙️

Now, let's define our model. It will have three layers:

  * **Embedding Layer**: Converts word indices into dense vectors (embeddings). This helps the model understand relationships between words.
  * **LSTM Layer (Long Short-Term Memory)**: Processes the sequence of embeddings, capturing context and order. It's a type of Recurrent Neural Network (RNN).
  * **Linear Layer**: Takes the output from the LSTM and maps it to a score for every word in our vocabulary, predicting the next word.

<!-- end list -->

```python
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleLLM, self).__init__()
        # Layer 1: Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Layer 2: LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # Layer 3: Linear (fully connected)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x is a tensor of word indices
        embedded = self.embedding(x)
        # The LSTM returns output and a tuple of (hidden_state, cell_state)
        lstm_out, _ = self.lstm(embedded)
        # We only want the output of the last time step
        last_hidden_state = lstm_out[:, -1, :]
        # Pass the last hidden state to the linear layer
        out = self.linear(last_hidden_state)
        return out

# Hyperparameters
EMBED_SIZE = 32
HIDDEN_SIZE = 64
```

-----

### \#\# 3. The Training Loop 🚀

This is where the model learns. We'll loop through our data many times (epochs), and for each input-target pair, we'll perform these steps:

1.  Get the model's prediction.
2.  Calculate the error (loss) using **CrossEntropyLoss**.
3.  Use backpropagation to calculate gradients.
4.  Update the model's weights with an **optimizer** like Adam.

<!-- end list -->

```python
# Instantiate the model, loss function, and optimizer
model = SimpleLLM(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# --- Training ---
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for input_seq, target_word in data:
        # Convert data to tensors
        input_tensor = torch.tensor([input_seq], dtype=torch.long)
        target_tensor = torch.tensor([target_word], dtype=torch.long)

        # 1. Forward pass
        output = model(input_tensor)

        # 2. Calculate loss
        loss = criterion(output, target_tensor)

        # 3. Backward pass and optimization
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data):.4f}')
```

-----

### \#\# 4. Text Generation (Inference)

After training, we can use our model to generate text. We give it a starting word (a "prompt"), and it will predict the next word. We then feed the new sequence back into the model to get the next word, and so on.

```python
def generate_text(model, start_word, max_length=10):
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # No need to calculate gradients during inference
        current_word = start_word.lower()
        generated_sentence = [current_word]
        
        # Prepare initial input
        input_seq = [word_to_idx[current_word]]
        
        for _ in range(max_length - 1):
            input_tensor = torch.tensor([input_seq], dtype=torch.long)
            
            # Get model output
            output = model(input_tensor)
            
            # Get the index of the word with the highest probability
            _, predicted_idx = torch.max(output, 1)
            predicted_word = idx_to_word[predicted_idx.item()]
            
            # Stop if we predict a word we don't know (though unlikely with this small vocab)
            if predicted_word not in word_to_idx:
                break
            
            # Append the predicted word and update the input sequence
            generated_sentence.append(predicted_word)
            input_seq.append(predicted_idx.item())

    return ' '.join(generated_sentence)

# --- Generate some text! ---
print("\n--- Text Generation ---")
prompt = "A"
generated = generate_text(model, prompt, max_length=10)
print(f"Prompt: '{prompt}' -> Generated: '{generated}'")

prompt = "Water"
generated = generate_text(model, prompt, max_length=8)
print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
```

This simple example covers the entire pipeline of building and training a language model in PyTorch. From here, you can explore more complex architectures like Transformers, larger datasets, and more advanced tokenization techniques.