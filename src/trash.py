# Model and Tokenizer Initialization

import torch
import transformers
from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = 'PY007/TinyLlama-1.1B-Chat-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model)

# Pipeline Initialization
pipeline = transformers.pipeline(
    'text-generation',
    model=model,
    torch_dtype=torch.float16,
    device_map=device,
)

# Prompt
prompt = 'What are the values in open source projects?'
formatted_prompt = f'### Human: {prompt}### Assistant:'

# Generate the Texts
sequences = pipeline(
    formatted_prompt,
    do_sample=True,
    top_k=50,
    top_p=0.7,
    num_return_sequences=1,
    repetition_penalty=1.1,
    max_new_tokens=500,
)

# Print the result
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
