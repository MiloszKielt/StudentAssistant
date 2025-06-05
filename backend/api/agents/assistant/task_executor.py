import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, TypedDict, Literal

from langchain_core.embeddings import Embeddings
from langgraph.graph import StateGraph, END
from langsmith import traceable

from backend.api.agents.RAG.rag_agent import RAGAgent
from backend.api.agents.assistant.decision_agent import ContextDecisionAgent
from backend.api.mcp_client import MCPClient
from backend.core.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class TaskState(TypedDict):
            num: int
            task: str
            context: str
            context_decision: str
            question_task: Optional[str]
            generated_questions: Optional[str]
            web_search_iterations: int

@dataclass
class TaskExecutor(BaseAgent):
    embedding_model: Embeddings
    documents_path: Optional[Union[str, Path]] = None
    
    def _create_agent(self):
        rag_agent = RAGAgent(self.llm, self.embedding_model, self.documents_path)
        ctx_decision_agent = ContextDecisionAgent(self.llm)
        mcp_client = MCPClient(f"http://localhost:{os.getenv('MCP_PORT', '8000')}")
        
        MAX_ITERATIONS = 1
            
        @traceable(name="RAG")
        async def rag_node(state: TaskState) -> TaskState:
            state['context'] = await rag_agent.ainvoke(state['task']) + '\n'
            
            return state
        
        @traceable(name="Web search")
        async def web_node(state: TaskState) -> TaskState:
            context = state['context']

            if context == 'No context available for this question.':
                context = ''
                
            logger.info(f"Calling mcp server for web search with context: {context}")
            state['context'] = context + await mcp_client.call_tool("search_web", {"query": state['task']})
                
            state['web_search_iterations'] += 1
            
            return state
        
        @traceable(name="Generate questions")
        async def question_node(state: TaskState) -> TaskState:
            ques = state['question_task']
            if ques == None:
                return state
            
            context = state['context']
                
            logger.info(f"Calling mcp server for question generation")
            state['generated_questions'] = await mcp_client.call_tool("create_exam_questions", {"query": ques, "context": context})
            
            return state
        
        #Conditions' routers
        def context_decision(state: TaskState) -> Literal['question_generation', 'web_search']:
            if state['web_search_iterations'] >= MAX_ITERATIONS:
                return 'question_generation'
        
            context = state['context']
            state['context_decisions'] = ctx_decision_agent.invoke(f"MESSAGE: {state['task']}\nCONTEXT: {context}")
                
            if state['context_decision'] == 'Yes' :
                return 'question_generation'
            else:
                return 'web_search'
        
        workflow = StateGraph(TaskState)
        workflow.set_entry_point('rag')
        workflow.add_node('rag', rag_node)
        workflow.add_node('web_search', web_node)
        workflow.add_node('question_generation', question_node)
        
        workflow.add_conditional_edges('rag', context_decision)
        workflow.add_conditional_edges('web_search', context_decision)
        workflow.add_edge('question_generation', END)
        
        return workflow.compile()
    