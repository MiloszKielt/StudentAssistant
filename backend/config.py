import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('study_assistant.log')  # File output
    ]
)

class Config:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-4o-mini"
    VECTOR_DB_DIR = "./storage/vector_db"
    UPLOAD_DIR = "../storage/uploads"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")