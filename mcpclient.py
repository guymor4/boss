from contextlib import AsyncExitStack
from dataclasses import dataclass
import sys
from typing import Optional
from mcp import Tool
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

class MCPClient:
    def __init__(self, name: str, server_url: str, bearer_token: str):
        self.name = name
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.server_url = server_url
        self.bearer_token = bearer_token
        self.tools: list[Tool] = []


    async def discover_tools(self) -> list[Tool]:
        async with sse_client(url=self.server_url, headers={'Authorization': f'Bearer {self.bearer_token}'}) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                self.session = session
                await self.session.initialize()

                response = await self.session.list_tools()
                self.tools = response.tools
                return self.tools

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool by name with the given arguments"""
        async with sse_client(url=self.server_url, headers={'Authorization': f'Bearer {self.bearer_token}'}) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                self.session = session
                await self.session.initialize()

                response = await self.session.call_tool(tool_name, arguments)
                if response.isError:
                    raise Exception(response.content[0].text)

    async def cleanup(self):
        """Clean up resources"""
        # if self.session:
        #     await self.session.close()