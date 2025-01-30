from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from datetime import datetime
from src.model_loader import ModelManager
import asyncio
from contextlib import asynccontextmanager


# Configuration

MODEL_NAMES = {
    1: 'mistralai/Mistral-7B-v0.1',
    2: 'distilbert/distilgpt2',
    3: 'gpt2',
    4: 'gpt2-medium',
    5: 'gpt2-large',
    6: 'tiiuae/falcon-rw-1b',
    7: 'PY007/TinyLlama-1.1B-step-50K-105b',
    8: 'NousResearch/Llama-2-7b-chat-hf',
    
}

MODEL_NAME = MODEL_NAMES[4]


# Global model manager
model_manager = ModelManager(MODEL_NAME)

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Application lifecycle management"""
    try:
        # Load model during startup
        await model_manager.load_model()
        await model_manager.start_processing()
        
        yield  
    finally:
        await model_manager.stop_processing()

app = FastAPI(
    title='Word Prediction Service',
    description='Async batch prediction service with model management',
    version='0.2',
    lifespan=app_lifespan
)

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    timestamp: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Process individual prediction request"""
    try:
        await model_manager.input_queue.put({'text': request.text})
        result = await model_manager.output_queue.get()
        
        return PredictionResponse(
            text=result['text'], 
            prediction=result['prediction'],
            timestamp=result['timestamp']
        )
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="Service overloaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "ok", "model": MODEL_NAME}