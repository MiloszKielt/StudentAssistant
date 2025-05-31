from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, ClassVar, TypedDict, List, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

from backend.core.agents.RAG.rag_agent import RAGAgent
from backend.core.agents.base_agent import BaseAgent
from backend.core.agents.exam_question_agent import ExamGenAgent
from backend.core.agents.web_search_agent import WebSearchAgent
from backend.core.agents.assistant.decision_agent import ContextDecisionAgent, QuestionDecisionAgent
from backend.core.agents.assistant.summarize_agent import SummarizeAgent
from backend.core.validation_methods import validate_string
from langchain_core.messages.ai import AIMessage
from langsmith import traceable

@dataclass
class AssistantAgent(BaseAgent):
    embedding_model: Embeddings
    documents_path: Optional[Union[str, Path]] = None
    tavily_max_results: int = 5
    
    @traceable
    def invoke(self, question: str):
        if not validate_string(question):
            raise ValueError("Question must be a valid nonempty string!")
        
        response = self.graph.invoke({'message': question, 'context': '', 'web_search_iterations': 0, 'questions': '', 'result': ''})
        return response['result']
    
    def _create_agent(self):
        rag_agent = RAGAgent(self.llm, self.embedding_model, self.documents_path)
        web_agent = WebSearchAgent(self.llm, self.tavily_max_results)
        exam_agent = ExamGenAgent(self.llm)
        ctx_decision_agent = ContextDecisionAgent(self.llm)
        q_decision_agent = QuestionDecisionAgent(self.llm)
        summarize_agent = SummarizeAgent(self.llm)
        
        MAX_ITERATIONS = 3
        
        class AssistantState(TypedDict):
            message: str
            context: str
            web_search_iterations: int
            questions: str
            result: str
            
        # Nodes
        @traceable(name="RAG")
        def rag_node(state: AssistantState) -> AssistantState:
            message = state['message']
            state['context'] += rag_agent.invoke(message) + '\n'
            
            return state
        
        @traceable(name="Web search")
        def web_node(state: AssistantState) -> AssistantState:
            message = state['message']
            context = state['context']
            
            if context == 'No context available for this question.':
                context = ''
            
            context += web_agent.invoke(message) + '\n'
            state['context'] = context
            state['web_search_iterations'] += 1
            
            return state
        
        @traceable(name="Generate questions")
        def question_node(state: AssistantState) -> AssistantState:
            message = state['message']
            context = state['context']
            
            state['questions'] = exam_agent.invoke(f"MESSAGE: {message}\nCONTEXT: {context}")
            
            return state
        
        @traceable(name="Summarize Results")
        def sumarize_node(state: AssistantState) -> AssistantState:
            message = state['message']
            context = state['context']
            questions = state['questions']
        
            state['result'] = summarize_agent.invoke(f"MESSAGE: {message}\nCONTEXT: {context}")
            
            if len(questions) > 0:
                state['result'] += f'\nQuestions: {questions}'
            
            return state
            
        #Conditions' routers
        def context_decision(state: AssistantState) -> Literal['question_decision_router', 'web_search']:
            if state['web_search_iterations'] >= MAX_ITERATIONS:
                return 'question_decision_router'
                
            message = state['message']
            context = state['context']
            
            decision = ctx_decision_agent.invoke(f"MESSAGE: {message}\nCONTEXT: {context}")
            if decision == 'Yes':
                return 'question_decision_router'
            else:
                return 'web_search'
            
        def question_decision(state: AssistantState) -> Literal['question_generation', 'summarize']:
            message = state['message']
            
            decision = q_decision_agent.invoke(message)
            if decision == 'Yes':
                return 'question_generation'
            else:
                return 'summarize'
                
        
        workflow = StateGraph(AssistantState)
        workflow.add_node('rag', rag_node)
        workflow.add_node('web_search', web_node)
        workflow.add_node('question_decision_router', lambda state:state)
        workflow.add_node('question_generation', question_node)
        workflow.add_node('summarize', sumarize_node)
        
        workflow.set_entry_point('rag')
        workflow.add_conditional_edges('rag', context_decision)
        workflow.add_conditional_edges('web_search', context_decision)
        workflow.add_conditional_edges('question_decision_router', question_decision)
        workflow.add_edge('question_generation', 'summarize')
        workflow.add_edge('summarize', END)
        
        return workflow.compile()