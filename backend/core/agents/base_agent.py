from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langgraph.graph.graph import CompiledGraph

from backend.core.validation_methods import validate_llm, validate_string


@dataclass
class BaseAgent(ABC):
    """Base class for all agents in the system.
    This class provides a common interface for agents that can invoke a language model (LLM) to answer questions.
    """
    llm: BaseChatModel
    prompt: str = field(init=False)
    graph: CompiledGraph = field(init=False)
    
    def __post_init__(self):
        if not validate_llm(self.llm):
            raise ValueError("LLM must support function calling!")
        self.graph = self._create_agent()
    
    @abstractmethod
    def _create_agent(self):
        """Create the agent's graph using the provided LLM.
        This method should be implemented by subclasses to define the specific behavior of the agent.
        """
        pass
    
    def invoke(self, question: str) -> Optional[str | List[str | Dict]]:
        """Invoke the agent with a question and return the response.

        Args:
            question (str): The question to ask the agent.

        Raises:
            ValueError: If the question is not a valid nonempty string.

        Returns:
            Optional[str | List[str | Dict]]: The response from the agent, which can be a string or a list of strings or dictionaries.
        """
        if not validate_string(question):
            raise ValueError("Question must be a valid nonempty string!")
        
        response = self.graph.invoke({'messages': [('user', question)]})
        return next((msg.content for msg in response['messages'] if isinstance(msg, AIMessage) and msg.content), None)
    
    async def ainvoke(self, question: str) -> Optional[str | List[str | Dict]]:
        """Asynchronously invoke the agent with a question and return the response.

        Args:
            question (str): The question to ask the agent.

        Raises:
            ValueError: If the question is not a valid nonempty string.

        Returns:
            Optional[str | List[str | Dict]]: The response from the agent, which can be a string or a list of strings or dictionaries.
        """
        if not validate_string(question):
            raise ValueError("Question must be a valid nonempty string!")
        
        response = await self.graph.ainvoke({'messages': [('user', question)]})
        return next((msg.content for msg in response['messages'] if isinstance(msg, AIMessage) and msg.content), None)