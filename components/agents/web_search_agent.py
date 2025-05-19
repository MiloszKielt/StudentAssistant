from typing import Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

from components.validation_methods import validate_llm

WEB_SEARCH_AGENT_PROMPT = """
You are an agent responsible for retrieving information from web using tavily and wikipedia related to the given question.
Provide from 3 to 7 paragrahs of relevant information.
If you can't find any relevant information respond 'No context available for this question.'
"""

def create_web_search_agent(llm: Union[LLM, BaseChatModel]):
    if not validate_llm(llm):
        raise ValueError("LLM must be of type LLM or BaseChatModel and support function calling!")
    
    tavily = TavilySearchResults(max_results=5)
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    return create_react_agent(llm, tools=[tavily, wikipedia], prompt=WEB_SEARCH_AGENT_PROMPT)