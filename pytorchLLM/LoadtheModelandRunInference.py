from ModelDefinitionandTokenizer import SimpleTokenizer
from ModelDefinitionandTokenizer import SimpleLLM
import torch
from SavingtheModel import VOCAB_SIZE,EMBED_DIM,tokenizer

# --- Step 1: Instantiate the model architecture ---
# You need a new instance of the same model class.
loaded_model = SimpleLLM(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
print("\nNew, untrained model instance created.")

# --- Step 2: Load the saved state_dict ---
# This loads the weights and biases from your file into the new model.
# Use map_location for loading a GPU-trained model onto a CPU.
loaded_model.load_state_dict(torch.load('simple_llm_state_elon.pth', map_location=torch.device('cpu')))
print("Saved weights have been loaded into the model.")

# --- Step 3: Set the model to evaluation mode ---
# This is crucial! It disables layers like Dropout or BatchNorm that
# behave differently during training and inference.
loaded_model.eval()
print("Model set to evaluation mode: model.eval()")

# --- Step 4: Prepare input data and run inference ---
your_data = ["dog", "cat", "tigger"]
your_data.append("test")
for i in range(5):
    your_data.append(i)

print(f"\nRunning inference on data: {your_data}")

# Use torch.no_grad() to disable gradient calculations.
# This makes inference faster and uses less memory.
with torch.no_grad():
    for word in your_data:
        print("word:")
        print(word)
        # a. Tokenize the input word to get its ID
        token_id = tokenizer.encode(word)

        # b. Convert the ID to a PyTorch tensor
        # The model expects a batch, so we add an extra dimension with unsqueeze(0)
        input_tensor = torch.tensor([token_id]).unsqueeze(0) # Shape: [1, 1]

        # c. Pass the tensor through the model
        logits = loaded_model(input_tensor)

        # d. Get the prediction
        # We find the index (token ID) with the highest score (logit)
        predicted_token_id = torch.argmax(logits, dim=1).item()

        # e. Decode the predicted token ID back to a word
        predicted_word = tokenizer.decode(predicted_token_id)

        print(f"Input: '{word}' -> Predicted next word might be: '{predicted_word}'")