from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, TypedDict, Literal, Dict
import logging
import os

from langchain_core.embeddings import Embeddings
from langgraph.graph import StateGraph, END
from langsmith import traceable

from backend.api.agents.RAG.rag_agent import RAGAgent
from backend.api.agents.assistant.decision_agent import ContextDecisionAgent
from backend.api.agents.assistant.summarize_agent import SummarizeAgent
from backend.api.agents.assistant.task_planner import TaskPlanner
from backend.core.agents.base_agent import BaseAgent
from backend.core.validation_methods import validate_string
from backend.api.mcp_client import MCPClient

logger = logging.getLogger(__name__)

@dataclass
class AssistantAgent(BaseAgent):
    embedding_model: Embeddings
    documents_path: Optional[Union[str, Path]] = None
    tavily_max_results: int = 5
    
    @traceable(name="Assistant Agent")
    def invoke(self, question: str):
        if not validate_string(question):
            raise ValueError("Question must be a valid nonempty string!")
        
        response = self.graph.invoke({
            'message': question, 
            'web_search_iterations': 0, 
            'result': '',
            'tasks_': {},
            'context_': {},
            'context_decisions_': {},
            'question_tasks_': {},
            'generated_questions_': {},
        })
        return response['result']
    
    @traceable(name="Assistant Agent")
    async def ainvoke(self, question: str):
        if not validate_string(question):
            raise ValueError("Question must be a valid nonempty string!")
        
        response = await self.graph.ainvoke({
            'message': question, 
            'web_search_iterations': 0, 
            'result': '',
            'tasks_': {},
            'context_': {},
            'context_decisions_': {},
            'question_tasks_': {},
            'generated_questions_': {},
        })
        return response['result']
    
    def _create_agent(self):
        task_planner = TaskPlanner(self.llm)
        rag_agent = RAGAgent(self.llm, self.embedding_model, self.documents_path)
        ctx_decision_agent = ContextDecisionAgent(self.llm)
        summarize_agent = SummarizeAgent(self.llm)
        mcp_client = MCPClient(f"http://localhost:{os.getenv('MCP_PORT', '8000')}")
        
        MAX_ITERATIONS = 3
        
        class AssistantState(TypedDict):
            message: str
            web_search_iterations: int
            result: str
            tasks_: Dict[int, str]
            context_: Dict[int, str]
            context_decisions_: Dict[int, str]
            question_tasks_: Dict[int, Optional[str]]
            generated_questions_: Dict[int, str]
            
        # Nodes
        @traceable(name="Task Planner")
        def task_planner_node(state: AssistantState) -> AssistantState:
            tasks_str = task_planner.invoke(state['message'])
            tasks = TaskPlanner.result_to_dict(tasks_str)
            
            for num, tasks in tasks.items():
                state['tasks_'][num] = tasks['MAIN']
                state['question_tasks_'][num] = tasks['QUES']
            
            return state
            
        
        @traceable(name="RAG")
        def rag_node(state: AssistantState) -> AssistantState:
            for num, task in state['tasks_'].items():
                state['context_'][num] = rag_agent.invoke(task) + '\n'
            
            return state
        
        @traceable(name="Web search")
        async def web_node(state: AssistantState) -> AssistantState:
            for num, task in state['tasks_'].items():
                if state['context_decisions_'][num] == 'Yes':
                    continue
                
                context = state['context_'].get(num, '')

                if context == 'No context available for this question.':
                    context = ''
                
                logger.info(f"Calling mcp server for web search with context: {context}")
                state['context_'][num] = context + await mcp_client.call_tool("search_web", {"query": task})
                
            state['web_search_iterations'] += 1
            
            return state
        
        @traceable(name="Generate questions")
        async def question_node(state: AssistantState) -> AssistantState:
            for num, ques in state['question_tasks_'].items():
                if not ques:
                    continue
                    
                context = state['context_'].get(num, '')
                
                logger.info(f"Calling mcp server for question generation")
                state['generated_questions_'][num] = await mcp_client.call_tool("create_exam_questions", {"query": ques})
            
            return state
        
        @traceable(name="Summarize Results")
        def sumarize_node(state: AssistantState) -> AssistantState:
            summary = ""
            questions = state['generated_questions_']
            
            for num, task in state['tasks_'].items():
                context = state['context_'].get(num, '')
                summary += '-'*10 + f' {task} ' + '-'*10 + '\n\n'
                summary += summarize_agent.invoke(f"MESSAGE: {task}\nCONTEXT: {context}") + '\n'
                question_set = questions.get(num, None)
                
                if question_set:
                    summary += f'\nGenerated questions: \n\n{question_set}'
                
                summary += '\n\n'
                
            state['result'] = summary
            return state
            
        #Conditions' routers
        def context_decision(state: AssistantState) -> Literal['question_generation', 'web_search']:
            if state['web_search_iterations'] >= MAX_ITERATIONS:
                return 'question_generation'
            
            for num, task in state['tasks_'].items():
                context = state['context_'].get(num, '')
                state['context_decisions_'][num] = ctx_decision_agent.invoke(f"MESSAGE: {task}\nCONTEXT: {context}")
                
            if all(decision == 'Yes' for decision in state['context_decisions_'].values()):
                return 'question_generation'
            else:
                return 'web_search'
                
        
        workflow = StateGraph(AssistantState)
        workflow.add_node('task_planner', task_planner_node)
        workflow.add_node('rag', rag_node)
        workflow.add_node('web_search', web_node)
        workflow.add_node('question_generation', question_node)
        workflow.add_node('summarize', sumarize_node)
        
        workflow.set_entry_point('task_planner')
        workflow.add_edge('task_planner', 'rag')
        workflow.add_conditional_edges('rag', context_decision)
        workflow.add_conditional_edges('web_search', context_decision)
        workflow.add_edge('question_generation', 'summarize')
        workflow.add_edge('summarize', END)
        
        return workflow.compile()