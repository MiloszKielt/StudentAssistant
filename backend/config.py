import os

class Config:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-4o-mini"
    VECTOR_DB_DIR = "./storage/vector_db"
    UPLOAD_DIR = "./storage/uploads"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")