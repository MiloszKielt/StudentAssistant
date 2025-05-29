from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, ClassVar

from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from backend.core.agents.RAG.vector_store import VectorStoreProvider
from backend.core.agents.base_agent import BaseAgent


@dataclass
class RAGAgent(BaseAgent):
    __DEFAULT_PROMPT: ClassVar[str] = """
    You are an agent responsible for providing detailed information for the given question.
    Use available tool 'document_lookup' to retrieve relevant question related context.
    If you don't know the answer respond 'No context available for this question.'
    """
    
    embedding_model: Embeddings
    documents_path: Optional[Union[str, Path]] = None
    vector_store: VectorStoreProvider = field(init=False)
    
    def __post_init__(self):
        self.vector_store = (VectorStoreProvider(self.embedding_model, documents_path=self.documents_path) if self.documents_path else VectorStoreProvider(self.embedding_model))
        if not hasattr(self, 'prompt') or self.prompt is None:
            self.prompt = self.__DEFAULT_PROMPT
            
        super().__post_init__()
    
    def _create_agent(self):
        @tool
        def document_lookup(question: str):
            """Retrieves relevant documents/information based on the given question.

            Args:
                question (str): given question
            """
            return self.vector_store.retriever.invoke(question)
        
        return create_react_agent(self.llm, tools=[document_lookup], prompt=self.prompt)