from fastapi import FastAPI
from src.routers import example, predict
from contextlib import asynccontextmanager
from src.routers import predict
from asyncio import Queue


@asynccontextmanager
async def lifespan(app):
    print('Loading model...')
    yield
    print('Freeing resources...')


app = FastAPI(
    title='word_prediction',
    description='Fill the description',
    version='0.1',
    lifespan=lifespan
)


# healtcheck
app.include_router(example.router)


# predict
app.include_router(predict.router)



@app.get("/health")
async def health_check():
    return {"status": "ok"}