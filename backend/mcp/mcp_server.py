import logging

from dotenv import load_dotenv
from jsonrpcserver import serve, method, Success, Error

from backend.config import Config
from backend.core.models_provider import LLMFactory
from backend.mcp.agents.exam_question_agent import ExamGenAgent
from backend.mcp.agents.web_search_agent import WebSearchAgent

load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('study_assistant.log')  # File output
    ]
)

logger = logging.getLogger(__name__)

web_agent = WebSearchAgent(LLMFactory.openai())
exam_agent = ExamGenAgent(LLMFactory.openai())

@method
def listTools():
    logger.info("listTools called")
    return Success([
        {
        "name": "search_web",
        "description": "Search the Internet and return 3 to 7 paragraphs-long answer of relevant information or the message saying 'no context available for this question'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
        },
        {
        "name": "create_exam_questions",
        "description": """Prepare as many as requested exam-style open questions for provided topic in this format:
            1. <question 1>
            2. <question 2>
            ...
            """,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }])

@method
def callTool(tool: str, args: dict):
    logger.info(f"callTool called with tool={tool!r}, args={args!r}")
    if tool == "search_web":
        # MUST use (message:str)
        q = args.get("query", "")
        try:
            answer = web_agent.invoke(q)
            logger.info(f"callTool returning Success(payload of length {len(answer)})")
            logger.debug(f"Payload: {answer}")
            return Success(answer)
        except Exception as e:
            logger.exception("Web search failed")
            return Error(2, f"Search failed: {e}")
    
    if tool == "create_exam_questions":
        q = args.get("query", "")
        c = args.get("context", "")
        try:
            answer = exam_agent.invoke(f"MESSAGE:\n{q}\n\nCONTEXT:{c}\n\n")
            logger.info(f"callTool returning Success(payload of length {len(answer)})")
            logger.debug(f"Payload: {answer}")
            return Success(answer)
        except Exception as e:
            logger.exception("Exam creation failed")
            return Error(2, f"Creation failed: {e}")
        
    return Error(1, f"Unsupported tool {tool!r}")
    

logger.info("Web search MCP server listening on http://localhost:4001")
serve(port=Config.MCP_PORT)