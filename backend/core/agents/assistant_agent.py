from pathlib import Path
from typing import Union

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from backend.core.agents.RAG.rag_agent import create_rag_agent
from backend.core.agents.exam_question_agent import create_exam_question_gen_agent
from backend.core.agents.web_search_agent import create_web_search_agent
from backend.core.validation_methods import validate_llm

ASSISTANT_PROMPT = """
You are an AI assistant that can answer questions on a given topic and also prepare exam-style questions and answers on that topic.
You are equipped with specific tools which allow you to properly respond to received question.
First familiarize yourself with the given topic using 'use_rag' tool. If you decide you need more information use 'use_web' tool to expand your knowledgebase related to given topic.

Optionally is user asks you to generate questions on related topic use tool 'use_exam_gen' with retrieved data on the topic as a context.
"""

def invoke(graph, question: str):
    response = graph.invoke({'messages': [('user', question)]})
    return next((msg.content for msg in response['messages'] if isinstance(msg, AIMessage) and msg.content), None)

def create_assistant_agent(llm: BaseChatModel, embedding_model: Embeddings, documents_path: Union[Path, str] = None):
    if not validate_llm(llm):
        raise ValueError("LLM must be of type LLM or BaseChatModel and support function calling!")
    
    if not isinstance(embedding_model, Embeddings):
        raise ValueError("Embedding model must be valid!")
    
    rag = create_rag_agent(llm, embedding_model)
    web_search = create_web_search_agent(llm)
    exam_generation = create_exam_question_gen_agent(llm)
    
    @tool
    def use_rag(question: str):
        """Retrieves context relevant to the question from locally stored documents.

        Args:
            question (str): given question

        """
        return invoke(rag, question)
    
    @tool
    def use_web(question: str):
        """Retrieves context relevant to the question from web search and/or wikipedia.

        Args:
            question (str): given question

        """
        return invoke(web_search, question)
    
    @tool
    def use_exam_gen(question: str):
        """Generates a set of exam-style questions and answers related to provied topic and its context.

        Args:
            question (str): given question and its related context

        """
        return invoke(exam_generation, question)
    
    return create_react_agent(llm, tools=[use_rag, use_web, use_exam_gen], prompt=ASSISTANT_PROMPT)
    