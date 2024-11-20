from fastapi import FastAPI

from .routers import example

# import uvicorn # for testing 


app = FastAPI(
    title='word_prediction',
    description='Fill the description',
    version='0.1',
)

app.include_router(example.router)
app.include_router(model.router)
