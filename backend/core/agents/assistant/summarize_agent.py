from dataclasses import dataclass
from typing import ClassVar

from langgraph.prebuilt import create_react_agent

from backend.core.agents.base_agent import BaseAgent


@dataclass
class SummarizeAgent(BaseAgent):
    __DEFAULT_PROMPT: ClassVar[str] = """
        You are an expert assistant tasked with providing clear, accurate, and concise summaries in response to user questions.
        You will recieve user's message and context.

        --- 
        INSTRUCTIONS:
        1. Write a concise, well-structured summary that answers the user's question.
        2. Do not repeat the question.
        3. Maintain a neutral, informative tone.
        4. If context is insufficient, state clearly: "The available information is not enough to provide a confident answer."
        """

    
    def __post_init__(self):
        if not hasattr(self, 'prompt') or self.prompt is None:
            self.prompt = self.__DEFAULT_PROMPT
            
        super().__post_init__()
    
    def _create_agent(self):
        return create_react_agent(self.llm, tools=[], prompt=self.prompt)
    