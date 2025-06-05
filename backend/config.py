import os

class Config:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-4o-mini"
    VECTOR_DB_DIR = "./storage/vector_db"
    UPLOAD_DIR = "../storage/uploads"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MCP_PORT = int(os.getenv("MCP_PORT", "4001"))
    MCP_HOST = os.getenv("MCP_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_HOST = os.getenv("API_HOST", "localhost")
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))
    FRONTEND_HOST = os.getenv("FRONTEND_HOST", "localhost")