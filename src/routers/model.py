from fastapi import APIRouter
from pydantic import BaseModel

from src.model_starter import predict_words
from typing import List


router = APIRouter(
    prefix='/model',
    tags=['model'],
    responses={404: {'description': 'Not found'}},
)


@router.get('/predict')
class PredictionRequest(BaseModel):
    """A class representing the request for word predictions.

    Attributes:
        text (str): The text input for which predictions will be generated.
    """

    text: str
    predictions: List[str]


@router.post(
    '/predict',
    summary='Predict words based on input text',
    description='Returns word predictions based on the provided input text.',
    response_model=PredictionRequest,
)
async def get_word_predictions(prefix: str):
    """Retrieves word predictions based on the given prefix.

    Args:
        prefix (str): The input string used to generate word predictions.

    Returns:
        dict: A dictionary containing the word predictions.
    """
    predictions = predict_words(prefix)
    return PredictionRequest(text= prefix, predictions=predictions)
