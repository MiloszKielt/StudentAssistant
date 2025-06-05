import uuid, aiohttp

class MCPClient:
    def __init__(self, url):
        self.url = url
    
    async def _rpc(self, method: str, params:dict=None) -> str | dict | list[dict]:
        """Internal method to perform a JSON-RPC call to the MCP server."

        Args:
            method (str): Description of the method to call.
            params (str, optional): Parameters to pass to the method. Defaults to None.

        Returns:
            str | dict | list[dict]: The response from the MCP server, which can be a string, a dictionary, or a list of dictionaries.
        """
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

    async def list_tools(self) -> list[dict]:
        """List all available tools on the MCP server.
        
        Returns:
            list[dict]: A list of tools, their descriptions and parameters that are available on the MCP server.
        """
        resp = await self._rpc("listTools", {})
        if isinstance(resp, dict) and "result" in resp:
            return resp["result"]
        raise RuntimeError(f"No result in listTools: {resp!r}")
        
    async def call_tool(self, name: str, args) -> str:
        """Call a tool on the MCP server with the given name and arguments.
        This method sends a request to the MCP server to execute a specific tool with the provided arguments.

        Args:
            name (str): The name of the tool to call.
            args (dict): A dictionary of arguments to pass to the tool.

        Raises:
            RuntimeError: If the response does not contain a "result" key, indicating that the tool call failed or was not executed properly.

        Returns:
            str: The result of the tool call, as returned by the MCP server.
        """
        resp = await self._rpc("callTool", {"tool": name, "args": args})
        if isinstance(resp, dict) and "result" in resp:
            return resp["result"]
        raise RuntimeError(f"No result in callTool: {resp!r}")

