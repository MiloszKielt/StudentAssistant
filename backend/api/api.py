import logging

from fastapi import UploadFile, HTTPException, FastAPI
from dotenv import load_dotenv

from backend.config import Config
from backend.api.agents.assistant.assistant_agent import AssistantAgent
from backend.core.models_provider import LLMFactory, EmbeddingFactory
from backend.api.data.query_message import QueryMessage
from backend.api.data.query_response import QueryResponse

load_dotenv()
logger = logging.getLogger(__name__)
app = FastAPI()
assistant = AssistantAgent(LLMFactory.openai(), EmbeddingFactory.openai())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('study_assistant.log')
    ]
)


@app.post("/upload")
async def upload(file: UploadFile):
    """Endpoint to upload a file for processing by the assistant agent.

    Args:
        file (UploadFile): The file to be uploaded.
    """
    logger.info(f"Received file upload: {file.filename}")
    file_path = f"{Config.UPLOAD_DIR}/{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(file.file.read())
    except:
        logger.error("Error while writing the file")
        
    logger.info("File upload successful")
    return {"status": "success", "file_path" : file_path}

@app.post("/query")
async def query(query_message: QueryMessage) -> QueryResponse:
    """Endpoint to process a query message and return a response.

    Args:
        query_message (QueryMessage): The query message containing the user's question.

    Raises:
        HTTPException: If an error occurs during processing.

    Returns:
        QueryResponse: The response containing the answer to the query.
    """
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