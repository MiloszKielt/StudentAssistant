import uuid, aiohttp

class MCPClient:
    def __init__(self, url):
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

