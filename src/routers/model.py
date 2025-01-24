from time import time, sleep
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import asyncio
from uuid import uuid4
import logging
from datetime import datetime
# import flash_attn
# from flash_attn.flash_attention import flash_attention


logging.basicConfig(level=logging.INFO)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

set_seed(42)


model = None
tokenizer = None
request_queue = asyncio.Queue()
batch_size = 5


MODEL_NAMES = {
    1: 'mistralai/Mistral-7B-v0.1',
    2: 'distilbert/distilgpt2',
    3: 'gpt2',
    4: 'gpt2-medium',
    5: 'gpt2-large',
    6: 'tiiuae/falcon-rw-1b'
}


number = 4  # Wybierz model z listy
model_name = MODEL_NAMES[number - 1]
print(f"Loading model: {model_name}")


# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=False, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16
# )


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=quantization_config,
    device_map=device,
)

model.eval()
model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token


def calculate_perplexity(loss):
    return torch.exp(loss)

# Funkcja do obliczania accuracy
def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    return correct / total

# def print_llm_params():
#     param_dict = {}
    
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Number of parameters: {total_params}")

#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             # add to param_dict 
#             param_dict[name] = param.shape
            
            
#             param_dict(name, param.shape
#             print (name, param.shape)
            
#     return 
    
    
def memory_usage():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024/1024} MB")



async def predict_words(messages: list[str], max_length=30):
    """Generates word predictions based on the input message.

    Args:
        message (str): The input string (prefix) to base predictions on.
        max_length (int, optional): The maximum number of tokens to generate. Default is 20.

    Returns:
        List[str]: A list of predicted words or partial words based on the input message.
    """
    
    results = []
    

    input = tokenizer(messages, padding=True, return_tensors='pt').to(device)


    with torch.no_grad():
        outputs = model.generate(
            **input,
            max_length=len(input['input_ids'][0]) + max_length,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            num_beams=5,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )


    



    # prediction = [tokenizer.decode(output[0], skip_special_tokens=True).split() for output in outputs]
    prediction = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    #   prediction = tokenizer.decode(output, skip_special_tokens=True).split() 

    for idx, p in enumerate(prediction):
        if p.startswith(messages[idx]):
            prediction[idx] = p[len(messages[idx]):].strip()  # Usunięcie wspólnej części wejścia

        
        
        results.append(prediction)
        # print(prediction)

        # memory_usage()
    
    
    # Calculate loss and accuracy for each prediction
    all_losses = []
    all_accuracies = []
    
    
    for idx, p in enumerate(prediction):
        if p.startswith(messages[idx]):
            prediction[idx] = p[len(messages[idx]):].strip()  # Usunięcie wspólnej części wejścia
        
        # Obliczanie straty (loss) dla wygenerowanego tekstu
        target_ids = tokenizer(messages[idx], return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=input['input_ids'].to(device)).logits

        # Obliczanie straty
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        perplexity = calculate_perplexity(loss)
        accuracy = calculate_accuracy(logits, target_ids)

        # Drukowanie metryk
        logging.info(f"Perplexity: {perplexity.item()}, Accuracy: {accuracy}")

        # Dodanie metryk do wyników
        all_losses.append(perplexity.item())
        all_accuracies.append(accuracy)
    
            # Średnia z metryk
        avg_perplexity = sum(all_losses) / len(all_losses) if all_losses else None
        avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else None

        logging.info(f"Average Perplexity: {avg_perplexity}")
        logging.info(f"Average Accuracy: {avg_accuracy}")
    
    
    return results




# function to process a batch of input data
async def process_batch(input_data: list[str]):
    """Processes queries in batches, assigning each a timestamp.

    Args:
        input_data (list): List of queries (prompts) to process.

    Returns:
        List of results containing predictions and timestamps for each query.
    """
    results = []

    # Process input data in batches of size `batch_size`
    while input_data:
        batch = []
        
        # Wait for up to 1 second to gather a batch
        start_time = time()
        while len(batch) < batch_size and time() - start_time < 1:
            if input_data:
                batch.append(input_data.pop(0))  # Pop one item from the input data
                
        # If there are still items to process, but the batch isn't full yet, we process the current batch
        if batch:
            predictions = await predict_words(batch)
            
            for query, prediction in zip(batch, predictions):
                results.append({
                    "input": query,
                    "prediction": prediction,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S:")+ f"{datetime.now().microsecond // 1000:03d}",
                    "memory_usage": torch.cuda.memory_allocated() / 1024 / 1024
                })

        # If the queue is empty, break the loop
        if not input_data:
            break

    return results