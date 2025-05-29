from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, ClassVar

from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from backend.core.agents.RAG.rag_agent import RAGAgent
from backend.core.agents.base_agent import BaseAgent
from backend.core.agents.exam_question_agent import ExamGenAgent
from backend.core.agents.web_search_agent import WebSearchAgent


@dataclass
class AssistantAgent(BaseAgent):
    __DEFAULT_PROMPT: ClassVar[str] = """
    You are an AI assistant that can answer questions on a given topic and also prepare exam-style questions and answers on that topic.
    You are equipped with specific tools which allow you to properly respond to received question.
    First familiarize yourself with the given topic using 'use_rag' tool. If you decide you need more information use 'use_web' tool to expand your knowledgebase related to given topic.

    Optionally if user asks you to generate questions on related topic use tool 'use_exam_gen' with retrieved data on the topic as a context.
    """
    
    embedding_model: Embeddings
    documents_path: Optional[Union[str, Path]] = None
    tavily_max_results: int = 5
    
    def __post_init__(self):
        if not hasattr(self, 'prompt') or self.prompt is None:
            self.prompt = self.__DEFAULT_PROMPT
            
        super().__post_init__()
    
    def _create_agent(self):
        rag_agent = RAGAgent(self.llm, self.embedding_model, self.documents_path)
        web_agent = WebSearchAgent(self.llm, self.tavily_max_results)
        exam_agent = ExamGenAgent(self.llm)
        
        @tool
        def use_rag(question: str) -> str:
            """Retrieves context relevant to the question from locally stored documents.

            Args:
                question (str): given question
            """
            return rag_agent.invoke(question)
        
        @tool
        def use_web(question: str) -> str:
            """Retrieves context relevant to the question from web search and/or wikipedia.

            Args:
                question (str): given question
            """
            return web_agent.invoke(question)
        
        @tool
        def use_exam_gen(question: str) -> str:
            """Generates a set of exam-style questions and answers related to provided topic and its context.

            Args:
                question (str): given question and its related context
            """
            return exam_agent.invoke(question)
        
        return create_react_agent(self.llm, tools=[use_rag, use_web, use_exam_gen], prompt=self.prompt)