from fastapi import APIRouter
from pydantic import BaseModel
from model_starter import predict_words

from typing import List

router = APIRouter(
    prefix='/model',
    tags=['model'],
    responses={404: {'description': 'Not found'}},
)




class PredictionRequest(BaseModel):
    text: str
    max_predictions: int = 5
    
    
@router.post(
    '/predict',
    summary='Predict words based on input text',
    description='Returns word predictions based on the provided input text.',
    response_model=PredictionResponse,
)

@router.get("/predict")
def get_word_predictions(prefix: str, max_predictions: int = 3):
    predictions = predict_words(prefix, max_predictions)
    return {"predictions": predictions}

