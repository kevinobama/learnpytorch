from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("./DialoGPT-small")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def chat_with_dialogpt(prompt):
    # Encode the prompt
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        inputs,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1
    )

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the original prompt from the response
    response = response.replace(prompt, "").strip()
    return response

# Your prompt
prompt = "who are you."
response = chat_with_dialogpt(prompt)

print(f"User: {prompt}")
print(f"DialoGPT: {response}")