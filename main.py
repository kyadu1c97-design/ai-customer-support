from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import get_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "AI Customer Support API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/chat")
def chat(query: Query):
    try:
        answer = get_response(query.question)
        return answer
    except Exception as e:
        return {"error": str(e)}  