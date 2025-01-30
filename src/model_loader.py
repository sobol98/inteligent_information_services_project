import logging
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

from datetime import datetime


from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)


BATCH_SIZE = 5
MAX_QUEUE_SIZE = 20
BATCH_WAIT_TIMEOUT = 5.0

class ModelManager:
    def __init__(
        self, 
        model_name: str, 
        batch_size: int = BATCH_SIZE, 
        max_queue_size: int = MAX_QUEUE_SIZE, 
        batch_wait_timeout: float = BATCH_WAIT_TIMEOUT
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.batch_wait_timeout = batch_wait_timeout
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
        # Queues and processing
        self.input_queue = asyncio.Queue(maxsize=max_queue_size)
        self.output_queue = asyncio.Queue()
        self.processing_tasks = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def load_model(self):
        """Load model during application startup"""
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=False, 
                bnb_4bit_quant_type='nf4', 
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                # quantization_config=quantization_config,
                device_map=self.device,
            )
            self.model.eval()
            self.model = torch.compile(self.model)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
    
    async def predict_batch(self, inputs: List[str], max_length: int = 30):
        """Generate predictions for a batch of inputs"""
        input_tokens = self.tokenizer(inputs, padding=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **input_tokens,
                max_length=input_tokens['input_ids'][0].size(0) + max_length,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
            )
        
        decoded_predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove input prefix from predictions
        return [
            pred[len(msg):].strip() if pred.startswith(msg) else pred 
            for msg, pred in zip(inputs, decoded_predictions)
        ]
    
    async def batch_processor(self):
        """Process batches from the input queue"""
        while True:
            batch = []
            try:
                # First item with timeout
                first_item = await asyncio.wait_for(
                    self.input_queue.get(), 
                    timeout=self.batch_wait_timeout
                )
                batch.append(first_item)
                
                # Collect additional items quickly
                while len(batch) < self.batch_size:
                    try:
                        item = await asyncio.wait_for(
                            self.input_queue.get(), 
                            timeout=0.2
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                if batch:
                    inputs = [item['text'] for item in batch]
                    # await asyncio.sleep(2)
                    # logging.info(f"Processing batch: {inputs}")
                    predictions = await self.predict_batch(inputs)
                    
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3] 
                                        
                    for item, prediction in zip(batch, predictions):
                        await self.output_queue.put({
                            **item,
                            'prediction': prediction,
                            'timestamp': current_time  # Add timestamp here 
                        })
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
    
    async def start_processing(self, num_processors: int = 2):
        """Start batch processing tasks"""
        self.processing_tasks = [
            asyncio.create_task(self.batch_processor()) 
            for _ in range(num_processors)
        ]
    
    async def stop_processing(self):
        """Stop batch processing tasks"""
        for task in self.processing_tasks:
            task.cancel()
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
