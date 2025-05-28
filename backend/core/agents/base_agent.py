from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langgraph.graph.graph import CompiledGraph

from backend.core.validation_methods import validate_llm, validate_string


@dataclass
class BaseAgent(ABC):
    llm: BaseChatModel
    prompt: str = field(init=False)
    graph: CompiledGraph = field(init=False)
    
    def __post_init__(self):
        if not validate_llm(self.llm):
            raise ValueError("LLM must support function calling!")
        self.graph = self._create_agent()
    
    @abstractmethod
    def _create_agent(self):
        pass
    
    def invoke(self, question: str) -> Optional[Union[str, List[Union[str, Dict]]]]:
        if not validate_string(question):
            raise ValueError("Question must be a valid nonempty string!")
        
        response = self.graph.invoke({'messages': [('user', question)]})
        return next((msg.content for msg in response['messages'] if isinstance(msg, AIMessage) and msg.content), None)
    
    async def ainvoke(self, question: str) -> Optional[Union[str, List[Union[str, Dict]]]]:
        if not validate_string(question):
            raise ValueError("Question must be a valid nonempty string!")
        
        response = await self.graph.ainvoke({'messages': [('user', question)]})
        return next((msg.content for msg in response['messages'] if isinstance(msg, AIMessage) and msg.content), None)