"""
unsloth_wip FastMCP Server
Module: server.py
Description: Functions for server operations

Granger standard MCP server implementation for unsloth_wip.
"""

from fastmcp import FastMCP
from .unsloth_wip_prompts import register_all_prompts
from .prompts import get_prompt_registry

# Initialize server
mcp = FastMCP("unsloth_wip")
mcp.description = "unsloth_wip - Granger spoke module"

# Register prompts
register_all_prompts()
prompt_registry = get_prompt_registry()

@mcp.prompt()
async def capabilities() -> str:
    """List all MCP server capabilities"""
    return await prompt_registry.execute("unsloth_wip:capabilities")

@mcp.prompt()
async def help(context: str = None) -> str:
    """Get context-aware help"""
    return await prompt_registry.execute("unsloth_wip:help", context=context)

@mcp.prompt()
async def quick_start() -> str:
    """Quick start guide for new users"""
    return await prompt_registry.execute("unsloth_wip:quick-start")

def serve():
    """Start the MCP server"""
    mcp.run(transport="stdio")

async def validate():
    """Validate server configuration"""
    result = await capabilities()
    assert "unsloth_wip" in result.lower()
    print(" Server validation passed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(validate())
    serve()
