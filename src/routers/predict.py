from fastapi import APIRouter
from pydantic import BaseModel
from src.routers.model import predict_words
from datetime import datetime
import asyncio
from src.routers.model import process_batch



router = APIRouter(
    prefix='/predict',
    tags=['predict'],
    responses={404: {'description': 'Word predictions not found'}},
)


class InputData(BaseModel):
    """A class representing the request for word predictions.

    Attributes:
        text (str): The text input for which predictions will be generated.
    """
    text: list[str]
    

class OutputData(BaseModel):
    """
    A class representing the response for word predictions.
    
    Attributes:
        predictions (list[str]): A list of predicted words or partial words based on the input text.
    
    """

    predictions: list[str]
    timestamp: str



@router.post(
    '/predict',
    summary='Predict words based on input text',
    description='Returns word predictions based on the provided input text.',
    response_model=OutputData,
)
async def get_word_predictions(input_data: InputData):
    """Retrieves word predictions based on the given text.

    Args:
        input_data (InputData): The input containing the text for which predictions are generated.

    Returns:
        OutputData: A response object containing the list of predictions.
    """
    
    # future = asyncio.Future()
    # request_data = {
    #     'data': input_data.text,
    #     'future': future
    # }
    
    predictions = []
    timestamp = None
    
    results = await process_batch(input_data.text)
    # print(input_data.text)
    
    for result in results:
        print(f"Processed input: {result['input']}, Prediction: {result['prediction']}, Time: {result['timestamp']}")

    if results:
        predictions = [str(item) for sublist in [result['prediction'] for result in results] for item in sublist]
        timestamp = results[0]['timestamp']  # Assuming timestamp is the same for all predictions

    
    # return OutputData(predictions=flattened_predictions, timestamp=timestamp)
    return OutputData(predictions=predictions, timestamp=timestamp)
