from dataclasses import dataclass
from typing import ClassVar

from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

from backend.core.agents.base_agent import BaseAgent


@dataclass
class WebSearchAgent(BaseAgent):
    __DEFAULT_PROMPT: ClassVar[str] = """
        You are an agent responsible for retrieving information from web using tavily and wikipedia related to the given question.
        Provide from 3 to 7 paragrahs of relevant information.
        If you can't find any relevant information respond 'No context available for this question.'
        """
    tavily_max_results: int = 5

    def __post_init__(self):
        if not hasattr(self, 'prompt') or self.prompt is None:
            self.prompt = self.__DEFAULT_PROMPT
            
        super().__post_init__()
    
    def _create_agent(self):
        tavily = TavilySearchResults(max_results=self.tavily_max_results)
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        
        return create_react_agent(self.llm, tools=[tavily, wikipedia], prompt=self.prompt)