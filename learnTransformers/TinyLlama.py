# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("./TinyLlama-1.1B-Chat-v1.0")
messages = [
    {"role": "user", "content": "learn AI?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))