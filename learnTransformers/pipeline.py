import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# https://modelscope.cn   ,   https://hf-mirror.com

from transformers import pipeline

pipe = pipeline("text-generation", model="openai-community/gpt2")
#pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B")

prompt = "learn AI"

# Generate a response
result = pipe(
    prompt,
    max_length=50,          # Maximum length of the generated text (including prompt)
    num_return_sequences=1  # Number of responses to generate
)

print(result)

# [{'generated_text': 'learn AI" is a concept that has been around since the first time a computer was used to solve problems. It\'s not a new phenomenon, but it\'s an important one, especially in the AI field. It\'s not just the workarounds that are lacking in today\'s artificial intelligence. It\'s the fact that it can\'t learn anything. It can\'t predict the future.\n\nFor instance, when you\'re asked to "learn to do this," you\'re going to say "This is really hard." It\'s not going to be easy. It\'s going to be a lot easier to learn. It\'s going to be much easier to learn than it was before.\n\nIn fact, the idea of a human being learning to do this kind of task is far from new. It was already in use in the past.\n\nAnd it\'s not just the workarounds. It\'s also the fact that it can\'t learn anything. It\'s not just the workarounds. It\'s also the fact that it can\'t predict the future. It\'s like a computer that can do only one thing.\n\nWhen you\'re asked to do something, you\'re going to say "That is so hard." It\'s not going to be easy. It\'s'}]
