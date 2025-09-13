# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiohttp",
#     "fastapi",
#     "pydantic",
#     "typer",
#     "uvicorn",
# ]
# ///
import asyncio
from functools import wraps
import json

import aiohttp
from attr import dataclass
from attr import dataclass
from mcp import Tool
from mcpclient import MCPClient
import sys
import typer
from typing import Optional

def mcp_tool_to_openai_format(mcp_tool: Tool) -> dict:
    """Convert MCP tool to OpenAI-compatible format for llama.cpp"""
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema or {"type": "object", "properties": {}}
        }
    }

@dataclass
class ToolCall:
    name: str
    arguments: dict

def typer_async_workaround():
    'Adapted from https://github.com/fastapi/typer/issues/950#issuecomment-2351076467'
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    return decorator

@typer_async_workaround()
async def main(
    prompt: str,
    endpoint: str = typer.Option('http://localhost:8080/v1/', '--endpoint', '-e', help='LLM endpoint URL', show_default=True),
    api_key: str = typer.Option(None, '--api-key', '-k', help='LLM API key, might not be required', show_default=False),
    mcp_server: Optional[str] = typer.Option(None, '--mcp-server', '-m', help='MCP server URL to discover tools from'),
    mcp_server_bearer_token: Optional[str] = typer.Option(None, '--mcp-bearer-token', '-b',
        help='Bearer token for MCP server authentication (or set MCP_BEARER_TOKEN environment variable)',
        envvar='MCP_BEARER_TOKEN', show_default=False),
    show_tools: bool = typer.Option(False, '--show-tools', '-s', help='Show discovered tools and exit'),
):

    available_tools: list[Tool] = []
    if mcp_server:
        assert mcp_server_bearer_token, 'MCP server bearer token is required when using MCP server'
        client = MCPClient('Home Assistant', mcp_server, mcp_server_bearer_token)
        try:
            print(f'Connecting to {client.name} MCP server...', end='', file=sys.stderr)
            tools = await client.discover_tools()
            available_tools.extend(tools)
            print(f' ✅ {len(tools)} tools available', file=sys.stderr)
        except Exception as e:
            print(' ❌ failed:', e, file=sys.stderr)
            raise
        finally:
            await client.cleanup()

    # If --show-tools is set, display the full tools and exit
    if show_tools:
        print('Discovered tools:', file=sys.stderr)
        if not available_tools:
            print(' (none)', file=sys.stderr)
        for tool in available_tools:
            print(f'- {tool.name}: {tool.description}', file=sys.stderr)
            print(f'  Parameters: {json.dumps(tool.inputSchema, indent=2)}', file=sys.stderr)
        return
    else:    
        print('Discovered tools:', [tool.name for tool in available_tools], file=sys.stderr)

    messages = [{
        'role': 'user',
        'content': prompt,
    }]
    headers = {
            'Content-Type': 'application/json',
        }
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    url = f'{endpoint}chat/completions'
    payload = {
        'model': 'gpt-4o',
        'messages': messages,
        'tools': [mcp_tool_to_openai_format(tool) for tool in available_tools],
    }
    
    tool_call: ToolCall | None = None
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            response = await response.json()
            print(f'Response: {json.dumps(response, indent=2)}', file=sys.stderr)
            tool_call = parse_tool_call_response(response)
    
    if not tool_call:
        print('No tool call detected in response')
        return
    
    print(f'Performing tool call: {tool_call.name} with arguments {json.dumps(tool_call.arguments, indent=2)}')
    await client.call_tool(tool_call.name, tool_call.arguments)

def parse_tool_call_response(response: aiohttp.ClientResponse) -> Optional[dict]:
    """Parse the tool call response from the LLM"""
    assert len(response['choices']) == 1
    choice = response['choices'][0]
    assert choice['finish_reason'] == 'tool_calls'
    tool_calls = choice['message']['tool_calls']
    assert tool_calls, 'No tool calls found in response'
    
    assert len(tool_calls) == 1, 'Only one tool call is supported currently'
    tool_call = tool_calls[0]
    
    # Parse tool call details
    return ToolCall(
        name=tool_call['function']['name'],
        arguments=json.loads(tool_call['function']['arguments'])
    )

if __name__ == '__main__':
    typer.run(main)
