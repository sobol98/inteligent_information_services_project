import torch
import torch.nn.functional as F
from transformers




# fine tuning TinyLlama model
# load the model
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load the dataset
from datasets import load_dataset
dataset = load_dataset('openwebtext')

# tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'])

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['text'])

# fine-tune the model
from transformers import Trainer, TrainingArguments

# define the training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
)

# define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# train the model
trainer.train()

# save the model
model.save_pretrained('TinyLlama-finetuned')

# save the tokenizer
tokenizer.save_pretrained('TinyLlama-finetuned')

# test the model
from transformers import pipeline
generator = pipeline('text-generation', model='TinyLlama-finetuned', tokenizer= 'TinyLlama-finetuned')
generator('Hello, how are you?')

# test the model
from transformers import pipeline

