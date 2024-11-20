from time import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# start time
start = time()

model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
# model_name='distilgpt2'

# model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# model = model.to(device)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map=device,
)


model.eval()
model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)


# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print (name, param.shape)

# print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024/1024} MB")


def predict_words(message: str, max_length=20):
    """Generates word predictions based on the input message.

    Args:
        message (str): The input string (prefix) to base predictions on.
        max_length (int, optional): The maximum number of tokens to generate. Default is 20.

    Returns:
        List[str]: A list of predicted words or partial words based on the input message.
    """
    if not message.endswith(' '):
        message += ' '

    input_ids = tokenizer(message, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + max_length,
            do_sample=True,
            top_p=0.85,
            top_k=30,
            temperature=1.5,
        )

    prediction = tokenizer.decode(output[0], skip_special_tokens=True).split()

    return prediction
    # return [word for word in prediction if word.startswith(message.strip())]

    # predictions = []
    # for gen_output in output:
    #     decoded = tokenizer.decode(gen_output, skip_special_tokens=True).split()
    #     predictions.append([word for word in decoded if word.startswith(prefix.strip())])

    # # Flatten and return the first `max_suggestions` words
    # return [word for sublist in predictions for word in sublist][:max_predictions]


# -------------------------------------------------------------------------------

prefix = 'mon'
num_test = 1

for _ in range(num_test):
    start_test = time()
    print(predict_words(prefix))
    end_test = time()
    print(f'Time taken for signle iteration (ms): {1000*(end_test-start_test)}')

# #end time


# end = time()
# print(f"Time taken (ms): {1000*(end-start)}")

# -------------------------------------------------------------------------------


# messages=' '
# inputs = tokenizer.encode(messages, return_tensors="pt", padding="max_length", max_length=20).to(device)
# # .to(device)
# print("Message tokens: ", inputs.shape)

# print(tokenizer.decode)

# outputs = model.generate(inputs, max_new_tokens=50, min_new_tokens=50)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
