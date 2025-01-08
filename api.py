from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import io

def chatbot_response(user_input: str) -> str:
    response = io(question=user_input)
    return f"Response: {response}"

app = FastAPI()

class ChatInput(BaseModel):
    message: str

@app.get("/chat")
async def chat(user_message: str):
    response = chatbot_response(user_message)
    return {"response": response}

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API!"}


# import uvicorn
# uvicorn.run(app, host="0.0.0.0", port=8080)

