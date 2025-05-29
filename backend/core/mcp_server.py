import logging
from jsonrpcserver import serve, method, Success, Error
from backend.core.agents.web_search_agent import WebSearchAgent
from backend.core.agents.exam_question_agent import ExamGenAgent

logging.basicConfig(level=logging.INFO)

web_agent = WebSearchAgent()
exam_agent = ExamGenAgent()

@method
def listTools():
    logging.info("→ listTools called")
    return Success([{
        "name": "search_web",
        "description": "Search the Internet and return 3 to 7 paragraphs-long answer of relevant information or the message saying 'no context available for this question'.",
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
    logging.info(f"→ callTool called with tool={tool!r}, args={args!r}")
    if tool != "search_web":
        # MUST use (message:str)
        return Error(1, f"Unsupported tool {tool!r}")
    q = args.get("query", "")
    try:
        answer = web_agent.ainvoke(q)
        logging.info(f"← callTool returning Success(payload of length {len(answer)})")
        logging.debug(f"Payload: {answer}")
        return Success(answer)
    except Exception as e:
        logging.exception("Web search failed")
        return Error(2, f"Search failed: {e}")
    

print("Web search MCP server listening on http://localhost:4001")
serve(port=4001)