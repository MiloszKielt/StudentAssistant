from typing import Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langgraph.prebuilt import create_react_agent

from components.validation_methods import validate_llm

EXAM_QUESTION_GENERATION_AGENT_PROMPT = """
You are an agent responsible for creating exam-style questions for the given topic.
Prepare as many as requested exam-style open questions for provided topic in this format:
    1. <question 1>
    2. <question 2>
    ...

After preparing the questions provide answers for each of them in following format:
    1. <answer for question 1>
    2. <answer for question 2>
    ...
"""

def create_exam_question_gen_agent(llm: Union[LLM, BaseChatModel]):
    if not validate_llm(llm):
        raise ValueError("LLM must be of type LLM or BaseChatModel and support function calling!")
    
    return create_react_agent(llm, tools=[], prompt=EXAM_QUESTION_GENERATION_AGENT_PROMPT)