#!/usr/bin/env python3
"""MCP server for Unsloth CLI.

This server exposes all Unsloth CLI commands via the Model Context Protocol,
allowing agents to interact with the training pipeline programmatically.
"""

import asyncio
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import click
from loguru import logger

try:
    from fastmcp import FastMCP
except ImportError:
    print("❌ FastMCP not installed!")
    print("\nInstall with: pip install fastmcp")
    sys.exit(1)


# Initialize MCP server
mcp = FastMCP("unsloth-mcp-server")


# ===== Core Training Commands =====

@mcp.tool()
async def unsloth_train(
    model: str,
    dataset: str,
    output: str = "./outputs/pipeline",
    hub_id: Optional[str] = None,
    force_runpod: bool = False,
    skip_enhancement: bool = False,
    max_samples: Optional[int] = None,
    grokking: bool = False
) -> Dict[str, Any]:
    """Run complete training pipeline (enhancement + training + upload).
    
    This automatically detects whether to use local or RunPod training based on model size.
    """
    cmd = [
        "unsloth", "train",
        "--model", model,
        "--dataset", dataset,
        "--output", output
    ]
    
    if hub_id:
        cmd.extend(["--hub-id", hub_id])
    if force_runpod:
        cmd.append("--force-runpod")
    if skip_enhancement:
        cmd.append("--skip-enhancement")
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    if grokking:
        cmd.append("--grokking")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout,
        "error": result.stderr if result.returncode != 0 else None,
        "command": " ".join(cmd)
    }


@mcp.tool()
async def unsloth_enhance(
    input: str,
    output: str,
    model: str,
    max_samples: Optional[int] = None,
    max_iterations: int = 3,
    batch_size: int = 10
) -> Dict[str, Any]:
    """Enhance Q&A dataset with student-teacher thinking.
    
    Uses the specified model as student and Claude as teacher to create
    iterative reasoning chains with self-correction.
    """
    cmd = [
        "unsloth", "enhance",
        "--input", input,
        "--output", output,
        "--model", model,
        "--max-iterations", str(max_iterations),
        "--batch-size", str(batch_size)
    ]
    
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout,
        "error": result.stderr if result.returncode != 0 else None
    }


@mcp.tool()
async def unsloth_validate(
    adapter: str,
    base_model: str,
    prompts: Optional[list] = None,
    compare_base: bool = False,
    dataset: Optional[str] = None
) -> Dict[str, Any]:
    """Validate a trained adapter."""
    cmd = [
        "unsloth", "validate",
        "--adapter", adapter,
        "--base-model", base_model
    ]
    
    if prompts:
        for prompt in prompts:
            cmd.extend(["--prompts", prompt])
    if compare_base:
        cmd.append("--compare-base")
    if dataset:
        cmd.extend(["--dataset", dataset])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout,
        "error": result.stderr if result.returncode != 0 else None
    }


# ===== RunPod Commands =====

@mcp.tool()
async def unsloth_runpod_list() -> Dict[str, Any]:
    """List all RunPod pods."""
    cmd = ["unsloth", "runpod", "list"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout,
        "error": result.stderr if result.returncode != 0 else None
    }


@mcp.tool()
async def unsloth_runpod_gpus() -> Dict[str, Any]:
    """List available GPUs on RunPod."""
    cmd = ["unsloth", "runpod", "gpus"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout,
        "error": result.stderr if result.returncode != 0 else None
    }


@mcp.tool()
async def unsloth_runpod_stop(pod_id: str, terminate: bool = False) -> Dict[str, Any]:
    """Stop a RunPod pod."""
    cmd = ["unsloth", "runpod", "stop", pod_id]
    if terminate:
        cmd.append("--terminate")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout,
        "error": result.stderr if result.returncode != 0 else None
    }


@mcp.tool()
async def unsloth_runpod_train(
    model: str,
    dataset: str,
    hub_id: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
) -> Dict[str, Any]:
    """Train a model on RunPod infrastructure."""
    cmd = [
        "unsloth", "runpod", "train",
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate)
    ]
    
    if hub_id:
        cmd.extend(["--hub-id", hub_id])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout,
        "error": result.stderr if result.returncode != 0 else None
    }


# ===== Utility Commands =====

@mcp.tool()
async def unsloth_quickstart() -> Dict[str, Any]:
    """Show quickstart guide."""
    cmd = ["unsloth", "quickstart"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "status": "success",
        "output": result.stdout
    }


@mcp.tool()
async def unsloth_models() -> Dict[str, Any]:
    """List recommended models and their requirements."""
    cmd = ["unsloth", "models"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "status": "success",
        "output": result.stdout
    }


@mcp.tool()
async def unsloth_upload(
    adapter: str,
    model_id: str,
    base_model: str,
    private: bool = True,
    tags: Optional[list] = None
) -> Dict[str, Any]:
    """Upload adapter to HuggingFace Hub."""
    cmd = [
        "unsloth", "upload",
        "--adapter", adapter,
        "--model-id", model_id,
        "--base-model", base_model
    ]
    
    if private:
        cmd.append("--private")
    if tags:
        for tag in tags:
            cmd.extend(["--tags", tag])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout,
        "error": result.stderr if result.returncode != 0 else None
    }


# ===== Server Configuration =====

@click.command()
@click.option("--host", default="localhost", help="Server host")
@click.option("--port", default=5555, type=int, help="Server port")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def main(host: str, port: int, debug: bool):
    """Run the Unsloth MCP server."""
    
    logger.info(f"🚀 Starting Unsloth MCP server on {host}:{port}")
    
    # List available tools
    tools = [
        "unsloth_train",
        "unsloth_enhance",
        "unsloth_validate",
        "unsloth_upload",
        "unsloth_runpod_list",
        "unsloth_runpod_gpus",
        "unsloth_runpod_stop",
        "unsloth_runpod_train",
        "unsloth_quickstart",
        "unsloth_models"
    ]
    
    logger.info(f"📋 Available tools: {len(tools)}")
    for tool in tools:
        logger.info(f"  - {tool}")
    
    # Run server
    try:
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped")


if __name__ == "__main__":
    main()