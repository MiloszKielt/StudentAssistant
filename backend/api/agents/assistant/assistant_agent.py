import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, TypedDict

from langchain_core.embeddings import Embeddings
from langgraph.graph import StateGraph, END
from langsmith import traceable

from backend.api.agents.assistant.summarize_agent import SummarizeAgent
from backend.api.agents.assistant.task_executor import TaskExecutor, TaskState
from backend.api.agents.assistant.task_planner import TaskPlanner
from backend.core.agents.base_agent import BaseAgent
from backend.core.validation_methods import validate_string

logger = logging.getLogger(__name__)

@dataclass
class AssistantAgent(BaseAgent):
    embedding_model: Embeddings
    documents_path: Optional[Union[str, Path]] = None
    
    @traceable(name="Assistant Agent")
    def invoke(self, question: str):
        if not validate_string(question):
            raise ValueError("Question must be a valid nonempty string!")
        
        response = self.graph.invoke({
            'message': question, 
            'result': '',
            'tasks': []
        })
        return response['result']
    
    @traceable(name="Assistant Agent")
    async def ainvoke(self, question: str):
        if not validate_string(question):
            raise ValueError("Question must be a valid nonempty string!")
        
        response = await self.graph.ainvoke({
            'message': question, 
            'result': '',
            'tasks': []
        })
        return response['result']
    
    def _create_agent(self):
        task_planner = TaskPlanner(self.llm)
        task_executor = TaskExecutor(self.llm, self.embedding_model, self.documents_path)
        summarize_agent = SummarizeAgent(self.llm)
        
        class AssistantState(TypedDict):
            message: str
            result: str
            tasks: list[TaskState]
            
        # Nodes
        @traceable(name="Task Planner")
        def task_planner_node(state: AssistantState) -> AssistantState:
            tasks_str = task_planner.invoke(state['message'])
            tasks = TaskPlanner.result_to_dict(tasks_str)
            
            for num, task in tasks.items():
                state['tasks'].append({
                    'num': num,
                    'task': task['MAIN'],
                    'context': '',
                    'context_decision': '',
                    'question_task': task['QUES'],
                    'generated_questions': '',
                    'web_search_iterations': 0
                })
            
            return state
            
        async def parallel_task_executor_node(state: AssistantState) -> AssistantState:
            """Implement to run TaskExecutor.graph.ainvoke(task) for each task in state[tasks]"""
            # Prepare all the task executions
            tasks_to_execute = []
            for task_state in state['tasks']:
                # Create a TaskState dictionary for each task
                task_input = {
                    'num': task_state['num'],
                    'task': task_state['task'],
                    'context': task_state['context'],
                    'context_decision': task_state['context_decision'],
                    'question_task': task_state['question_task'],
                    'generated_questions': task_state['generated_questions'],
                    'web_search_iterations': task_state['web_search_iterations']
                }
                # Add the task execution to the list
                tasks_to_execute.append(task_executor.graph.ainvoke(task_input))
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks_to_execute)
            
            # Update the state with the results
            for i, result in enumerate(results):
                state['tasks'][i].update(result)
            
            return state
        
        
        @traceable(name="Summarize Results")
        def sumarize_node(state: AssistantState) -> AssistantState:
            summary = ""
            
            for task in state['tasks']:
                context = task['context']
                summary += '-'*10 + f' {task["task"]} ' + '-'*10 + '\n\n'
                summary += summarize_agent.invoke(f"MESSAGE: {task['task']}\nCONTEXT: {context}") + '\n'
                questions = task['generated_questions']
                
                if questions:
                    summary += f'\nGenerated questions: \n\n{questions}'
                
                summary += '\n\n'
                
            state['result'] = summary
            return state
            
        
                
        
        workflow = StateGraph(AssistantState)
        workflow.add_node('task_planner', task_planner_node)
        workflow.add_node('parallel_task_executor', parallel_task_executor_node)
        workflow.add_node('summarize', sumarize_node)
        
        workflow.set_entry_point('task_planner')
        workflow.add_edge('task_planner', 'parallel_task_executor')
        workflow.add_edge('parallel_task_executor', 'summarize')
        workflow.add_edge('summarize', END)
        
        return workflow.compile()