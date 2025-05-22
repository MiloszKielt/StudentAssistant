from pathlib import Path
from typing import Union, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from backend.core.agents.RAG.vector_store import VectorStoreProvider
from backend.core.validation_methods import validate_string, validate_llm

RAG_PROMPT = """
You are an agent responsible for providing detailed information for the given question.
Use available tool 'document_lookup' to retrieve relevant question related context.
If you don't know the answer respond 'No context available for this question.'
"""

def create_rag_agent(llm: BaseChatModel, embedding_model: Embeddings, documents_path: Optional[Union[str, Path]] = None):
    if not validate_llm(llm):
        raise ValueError("LLM must be of type LLM or BaseChatModel and support function calling!")
    
    if not isinstance(embedding_model, Embeddings):
        raise ValueError("Embedding model must be valid!")
    
    if documents_path is not None:
        if not isinstance(documents_path, Path):
            if validate_string(documents_path):
                documents_path = Path(documents_path)
            else:  
                raise ValueError("Documents path must be provided as a vaid Path object or nonempty string!")
     
    vsp = VectorStoreProvider(embedding_model) if not documents_path else VectorStoreProvider(embedding_model, documents_path=documents_path)
    
    @tool
    def document_lookup(question: str):
        """Retrieves relevant documents/information based on the given question.

        Args:
            question (str): given question
        """
        return vsp.retriever.invoke(question)
    
    return create_react_agent(llm, tools=[document_lookup], prompt=RAG_PROMPT)