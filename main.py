# main.py
# Import necessary libraries
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
# Configure the Gemini API with your key
# It's best practice to store your API key in an environment variable
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # This will be caught by the exception handler in the API endpoint
        print("Error: GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
except Exception as e:
    # This will likely be a configuration error.
    print(f"Error configuring Gemini API: {e}")
    
# --- Model Initialization ---
# CHANGED: Switched to gemini-1.5-flash-latest. 
# This model is faster and has a more generous free tier for chat applications.
model = genai.GenerativeModel('gemini-1.5-flash-latest')
chat = model.start_chat(history=[])

# --- FastAPI App Initialization ---
app = FastAPI()

# Mount a directory for static files (like CSS or client-side JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup for serving HTML templates
templates = Jinja2Templates(directory="templates")

# --- Pydantic Models for Request Body ---
# This ensures the incoming request data has the expected structure
class ChatRequest(BaseModel):
    message: str

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML page for the chatbot.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_class=JSONResponse)
async def handle_chat(chat_request: ChatRequest):
    """
    Handles the chat logic. Receives a message from the user,
    sends it to the Gemini API, and returns the model's response.
    """
    try:
        # Send the message to the Gemini model
        # The 'stream=True' option can be used for a typing effect,
        # but for a simple JSON response, we'll get the full response at once.
        response = chat.send_message(chat_request.message)
        
        # Return the model's response text
        return {"reply": response.text}
    except Exception as e:
        # Handle potential exceptions from the API call, including auth/quota issues
        print(f"An error occurred during the API call: {e}")
        raise HTTPException(status_code=500, detail="Failed to get a response from the AI model. Check server logs for details.")

# To run this app:
# 1. Make sure you have a .env file with your GEMINI_API_KEY.
# 2. Create folders: `templates` and `static`.
# 3. Put the `index.html` file in the `templates` folder.
# 4. Run in your terminal: uvicorn main:app --reload
