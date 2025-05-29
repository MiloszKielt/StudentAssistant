import os
import logging

from fastapi import UploadFile, HTTPException, FastAPI
from pydantic import BaseModel

from config import Config
from core.agents.assistant_agent import AssistantAgent
from core.models_provider import LLMFactory, EmbeddingFactory
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
app = FastAPI()
assistant = AssistantAgent(LLMFactory.openai(), EmbeddingFactory.openai())
# ingestor = FileIngestor()


@app.post("/upload")
async def upload(file: UploadFile):
    logger.info(f"Received file upload: {file.filename}")
    file_path = f"{Config.UPLOAD_DIR}/{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(file.file.read())
    except:
        logger.error("Error while writing the file")
    # ingestor.ingest(file_path)
    logger.info("File upload successful")
    return {"status": "success", "file_path" : file_path}


class QueryMessage(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    answer: str

@app.post("/query")
async def query(query_message: QueryMessage) -> QueryResponse:
    logger.info("Incoming query...")
    try:
        response = await assistant.ainvoke(query_message.query)
        logger.info("Returning answer")
        return QueryResponse(answer=response)
    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Retrieval failed: {str(e)}"
        )