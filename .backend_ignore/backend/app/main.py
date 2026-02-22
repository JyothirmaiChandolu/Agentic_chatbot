"""
FastAPI main application entry point — MHK-GPT v2 Agentic System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create app
app = FastAPI(
    title="MHK-GPT v2 — Agentic AI Assistant",
    description="AI-powered agentic chatbot with RAG, Meeting Scheduling, and Job Search.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "message": "Backend is running!"}

@app.get("/")
async def root():
    return {"message": "Welcome to MHK-GPT Agentic Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
