import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

# Tworzymy przykładowy dataset z prostymi zdaniami
data = [
    {"text": "Hello, how are you?"},
    {"text": "I am learning AI."},
    {"text": "This is a test sentence."},
    {"text": "GPT-3 is powerful."},
    {"text": "Torch is great for deep learning."},
    {"text": "How do I install Python?"},
    {"text": "Machine learning is the future."},
    {"text": "I love programming with Python."},
    {"text": "Natural language processing is fun."},
    {"text": "Deep learning models are fascinating."}
]

# Konwertujemy na format datasetu
dataset = Dataset.from_dict({"text": [item["text"] for item in data]})

# Tokenizer i model GPT2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Funkcja tokenizująca dane
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=50)

# Tokenizacja
dataset = dataset.map(tokenize_function, batched=True)

# Podział na dane treningowe i testowe
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()

# DataLoader dla trenowania
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Funkcja obliczania perplexity
def calculate_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(model.device)
            labels = inputs.clone()
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            total_tokens += inputs.size(1)

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity

# Funkcja obliczania accuracy
def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(model.device)
            labels = inputs.clone()
            outputs = model(input_ids=inputs, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    accuracy = correct / total
    return accuracy

# Wyznaczanie perplexity i accuracy
perplexity = calculate_perplexity(model, train_loader)
accuracy = calculate_accuracy(model, train_loader)

print(f"Perplexity: {perplexity}")
print(f"Accuracy: {accuracy}")
