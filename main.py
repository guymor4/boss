import asyncio
from functools import wraps
import json
import time

import aiohttp
from attr import dataclass
from attr import dataclass
from mcp import Tool
from mcpclient import MCPClient
import sys
import typer
from typing import Optional

@dataclass
class ToolCall:
    name: str
    arguments: dict

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

def follow(filename):
    """Generator function that yields new lines added to a file (like tail -f)"""
    with open(filename, "r") as f:
        # seek to end of file
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line.strip()

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
    prompt: str = typer.Argument(..., help='The prompt to send to the LLM, "-" to read from stdin continuously'),
    endpoint: str = typer.Option('http://localhost:8080/v1/', '--endpoint', '-e', help='LLM endpoint URL', show_default=True),
    api_key: str = typer.Option(None, '--api-key', '-k', help='LLM API key, might not be required', show_default=False),
    mcp_server: Optional[str] = typer.Option(None, '--mcp-server', '-m', help='MCP server URL to discover tools from'),
    mcp_server_bearer_token: Optional[str] = typer.Option(None, '--mcp-bearer-token', '-b',
        help='Bearer token for MCP server authentication (or set MCP_BEARER_TOKEN environment variable)',
        envvar='MCP_BEARER_TOKEN', show_default=False),
    show_tools: bool = typer.Option(False, '--show-tools', '-s', help='Show discovered tools and exit'),
    dry_run: bool = typer.Option(False, '--dry-run', '-d', help='Do not perform any tool calls, just show what would be done'),
):
    think = True
    if dry_run:
        print('*Dry run mode enabled, no tool calls will be performed*', file=sys.stderr)
    available_tools: list[Tool] = []
    if think:
        available_tools.append(Tool(
            name='think',
            description='Call this function at every step to explain your thought process, before taking any other action',
            inputSchema={
                'type': 'object',
                'properties': {
                    'thoughts': {
                        'type': 'string'
                    },
                },
                'required': ['thoughts'],
            }
        ))
        
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

    if prompt == '-':
        print('Reading prompts from stdin...', file=sys.stderr)
        for line in sys.stdin:
            if not line:
                continue
            print(f'Processing prompt: {line.strip()}', file=sys.stderr)
            await process_prompt(line, endpoint, api_key, client, available_tools, dry_run)
    else:
        await process_prompt(prompt, endpoint, api_key, client, available_tools, dry_run)
    

async def process_prompt(prompt: str, llm_endpoint: str, llm_api_key: Optional[str], mcpClient: MCPClient, available_tools: list[Tool], dry_run: bool):
    headers = {
        'Content-Type': 'application/json',
    }
    if llm_api_key:
        headers['Authorization'] = f'Bearer {llm_api_key}'
        
    payload = {
        'messages': [{
            'role': 'system',
            'content': 'You are a helpful assistant that can call tools to perform actions on behalf of the user.',
        },
        {
            'role': 'system',
            'content': 'Available entities: "TV"',
        },
        {
            'role': 'user',
            'content': prompt,
        }],
        'tools': [mcp_tool_to_openai_format(tool) for tool in available_tools],
        'tool_choice': 'required',
        'verbosity': 'high',
    }
    
    tool_call: ToolCall | None = None
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(f'{llm_endpoint}chat/completions', json=payload) as response:
            response.raise_for_status()
            response = await response.json()
            # Verbose
            # print(f'Response: {json.dumps(response, indent=2)}', file=sys.stderr)
            tool_call = parse_tool_call_response(response)
    
    if not tool_call:
        print('No tool call detected in response')
        return
    
    if dry_run:
        return
    if tool_call.name == 'think':
        print(f'🧠 {tool_call.arguments["thoughts"]}', file=sys.stderr)
        return
    print(f'Performing tool call: {tool_call.name} with arguments {json.dumps(tool_call.arguments, indent=2)}')
    try:
        await mcpClient.call_tool(tool_call.name, tool_call.arguments)
        print('Tool call completed successfully')
    except Exception as e:
        print(f'Failed calling tool: {e}', file=sys.stderr)

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
