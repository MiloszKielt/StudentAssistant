import os

from fastapi import UploadFile, HTTPException, FastAPI
from pydantic import BaseModel

from config import Config
from core.file_ingestor import FileIngestor

app = FastAPI()
ingestor = FileIngestor()


@app.post("/upload")
async def upload(file: UploadFile):
    file_path = f"{Config.UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    ingestor.ingest(file_path)
    return {"status": "success", "file_path" : file_path}


class QueryMessage(BaseModel):
    query: str

@app.post("/query")
async def query(query: QueryMessage):
    try:
        if not os.path.exists(Config.VECTOR_DB_DIR):
            raise HTTPException(
                status_code=404,
                detail="No documents found. Upload files first."
            )
        
        # Retrieve relevant chunks
        docs = rag_engine.retrieve(query.query)
        
        # Format response
        return {
            "query": query,
            "results": [{"content": doc.page_content} for doc in docs]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )