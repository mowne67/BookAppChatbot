from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import io

# Your chatbot functions go here
def chatbot_response(user_input: str) -> str:
    # Replace this with your actual chatbot logic
    response = io(question=user_input)
    return f"Response: {response}"

# Create a FastAPI app
app = FastAPI()

# Define a Pydantic model for the input
class ChatInput(BaseModel):
    message: str

# API endpoint for the chatbot
@app.get("/chat")
async def chat(user_message: str):
    response = chatbot_response(user_message)
    return {"response": response}

# Add a simple GET endpoint for status
@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API!"}


import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8080)

