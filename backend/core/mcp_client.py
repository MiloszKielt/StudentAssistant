import uuid, aiohttp
import asyncio
import json
import openai
import os 
import dotenv


dotenv.load_dotenv(".env")  # Load environment variables from .env file
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-...")  # Set your OpenAI API key here


class MCPClient:
    def __init__(self, url="http://localhost:5000"):
        self.url = url

    async def _rpc(self, method, params=None):
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {}
        }
        async with aiohttp.ClientSession() as sess:
            async with sess.post(self.url, json=payload) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except:
                    data = text
        return data

    async def list_tools(self):
        resp = await self._rpc("listTools", {})
        if isinstance(resp, dict) and "result" in resp:
            return resp["result"]
        raise RuntimeError(f"No result in listTools: {resp!r}")

    async def call_tool(self, name, args):
        resp = await self._rpc("callTool", {"tool": name, "args": args})
        if isinstance(resp, dict) and "result" in resp:
            return resp["result"]
        raise RuntimeError(f"No result in callTool: {resp!r}")


mcp = MCPClient("http://localhost:4001")


TOOL_SPECS = asyncio.run(mcp.list_tools())

def to_openai_functions(toolspecs):
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]
        }
        for t in toolspecs
    ]
    
OPENAI_FUNCTIONS = to_openai_functions(TOOL_SPECS)
print(json.dumps(OPENAI_FUNCTIONS, indent=2))

async def chat_with_tools(user_message):
    # Initial agent definition
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that can call tools via MCP."},
            {"role": "user",   "content": user_message}
        ],
        functions=OPENAI_FUNCTIONS,
        function_call="auto"
    )
    msg = resp.choices[0].message

    # Let's see if the model decided to call a function
    if msg.function_call is not None:
        fn_name = msg.function_call.name
        fn_args = json.loads(msg.function_call.arguments)
        print(f"Agent asks for {fn_name}{fn_args}")

        # Use the tool via MCP server
        result = await mcp.call_tool(fn_name, fn_args)

        # Feed the result back in to get the final answer
        follow = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",  "content": "Return the final answer to the user."},
                {
                    "role": "assistant",
                    "function_call": {
                        "name": fn_name,
                        "arguments": msg.function_call.arguments
                    }
                },
                {
                    "role": "function",
                    "name": fn_name,
                    "content": json.dumps(result)
                }
            ]
        )
        return follow.choices[0].message.content

    return msg.content

async def main():
    res = await chat_with_tools("Search for the latest news about AI and summarize it in 3-5 sentences. Then create a short 4 question exam out of it")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())