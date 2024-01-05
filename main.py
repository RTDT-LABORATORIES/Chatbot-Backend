import sys
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from chatbot import Chatbot
from setup import setup

load_dotenv()
setup()

app = FastAPI()
origins = [
    "http://localhost:3000",  # Local front-end server
    "https://chat.rtdt.ai",  # Production front-end domain
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins (or use ["*"] for all)
    allow_credentials=True,  # Allow credentials (cookies, authorization headers)
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class Prompt(BaseModel):
    prompt: str
    session: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prompt")
def prompt(prompt: Prompt):
    return Chatbot(session=prompt.session).run(prompt.prompt)


if __name__ == "__main__":
    try:
        prompt = sys.argv[1]
    except IndexError:
        print("Invalid arguments: prompt is missing")

    Chatbot().run(prompt)
