from dataclasses import dataclass
from typing import ClassVar

from langgraph.prebuilt import create_react_agent

from backend.core.agents.base_agent import BaseAgent


@dataclass
class ExamGenAgent(BaseAgent):
    __DEFAULT_PROMPT: ClassVar[str] = """
        You are an agent responsible for creating exam-style questions for the given topic.
        You will recieve both user's message (topic) and its relevant context.
        Prepare as many as requested exam-style open-ended and close-ended questions for provided topic in this format:
            1. <question 1>
            2. <question 2>
            ...

        After preparing the questions provide answers for each of them in following format:
            1. <answer for question 1>
            2. <answer for question 2>
            ...
        """


    def __post_init__(self):
        if not hasattr(self, 'prompt') or self.prompt is None:
            self.prompt = self.__DEFAULT_PROMPT
            
        super().__post_init__()
    
    def _create_agent(self):
        return create_react_agent(self.llm, tools=[], prompt=self.prompt)