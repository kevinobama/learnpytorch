from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./TinyStories-1M"  # change to your actual folder path

# Load tokenizer and model from the local folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Your prompt
prompt = "Lion"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    **inputs,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    num_return_sequences=1
)

# Decode & print
print(tokenizer.decode(output[0], skip_special_tokens=True))
