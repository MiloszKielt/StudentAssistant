import re
from dataclasses import dataclass
from typing import ClassVar, Dict, Literal, Optional

from langgraph.prebuilt import create_react_agent

from backend.core.agents.base_agent import BaseAgent
from backend.core.validation_methods import validate_string


@dataclass
class TaskPlanner(BaseAgent):
    __DEFAULT_PROMPT: ClassVar[str] = """
        You are a task analyzer. Your role is to identify and separate two types of tasks from user input:
            The primary/main task (MAIN)
            Any related exam-style question generation requests (QUES)

        Instructions:
            Analyze the user's input carefully
            Identify and number all MAIN tasks sequentially (1, 2, 3...)
            Identify if MAIN tasks are related to another. If so properly format them
            For each MAIN task, check if it contains or implies a question generation request
            Only output a QUES line when there's an explicit exam question request
            Maintain identical numbering between related MAIN/QUES pairs
            If multiple QUES exist for one MAIN, use sub-numbering (QUES 3.1, QUES 3.2)
            Insert the topic into QUES.

            Output format examples:
                MAIN 1: <primary task without questions>
                MAIN 2: <primary task>
                QUES 2: <related exam questions request>
                MAIN 3: <another primary task>
                QUES 3.1: <first question request for MAIN 3>
                QUES 3.2: <second question request for MAIN 3>

            
            NOT create a QUES without a corresponding MAIN.
            NOT separate tasks by new line.
        """

    
    def __post_init__(self):
        if not hasattr(self, 'prompt') or self.prompt is None:
            self.prompt = self.__DEFAULT_PROMPT
            
        super().__post_init__()
    
    @staticmethod
    def result_to_dict(tasks: str) -> Dict[int, Dict[Literal['MAIN', 'QUES'], Optional[str]]]:
        if not validate_string(tasks):
            raise ValueError()
        
        lines = tasks.split('\n')
        entries = {}
        
        main_pattern = re.compile(r'^MAIN (\d+): (.+)$')
        ques_pattern = re.compile(r'^QUES (\d+): (.+)$')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            main_match = main_pattern.match(line)
            if main_match:
                num = int(main_match.group(1))
                text = main_match.group(2)
                entries[num] = {"MAIN": text, "QUES": None}
                
            ques_match = ques_pattern.match(line)
            if ques_match:
                num = int(ques_match.group(1))
                text = ques_match.group(2)
                if num in entries:
                    entries[num]["QUES"] = text
                else:
                    entries[num] = {"MAIN": None, "QUES": text}
        
        # result = [entries[key] for key in sorted(entries.keys(), key=int)]
        return entries
    
    def _create_agent(self):
        return create_react_agent(self.llm, tools=[], prompt=self.prompt)
    