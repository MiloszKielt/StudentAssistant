from typing import ClassVar

from langgraph.prebuilt import create_react_agent

from backend.core.agents.base_agent import BaseAgent


class ContextDecisionAgent(BaseAgent):
    __DEFAULT_PROMPT: ClassVar[str] = """
    You are an expert in decision-making.
    You will be given a message and its related context.  
    Determine whether the provided context is sufficient to understand or respond to the message.

    If the context is sufficient, reply with "Yes".  
    If it is not, reply with "No".
    """
    
    def __post_init__(self):
        if not hasattr(self, 'prompt') or self.prompt is None:
            self.prompt = self.__DEFAULT_PROMPT
            
        super().__post_init__()
    
    def _create_agent(self):
        return create_react_agent(self.llm, tools=[], prompt=self.prompt)