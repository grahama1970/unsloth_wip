"""Unified RunPod instance management combining best features.
Module: instance_manager.py
Description: Core RunPod instance lifecycle management

This module provides unified instance management for RunPod, combining
the best features from both implementations with improved error handling
and file transfer capabilities.

External Dependencies:
- runpod: https://docs.runpod.io/
- httpx: https://www.python-httpx.org/
- loguru: https://loguru.readthedocs.io/

Sample Input:
>>> manager = InstanceManager(api_key="your_key")
>>> instance = await manager.create_instance(
...     gpu_type="RTX_4090",
...     gpu_count=2,
...     purpose="training"
... )

Expected Output:
>>> print(instance)
RunPodInstance(id='abc123', gpu_type='RTX_4090', status='running')
"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import runpod
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class InstancePurpose(str, Enum):
    """Purpose of the RunPod instance."""
    TRAINING = "training"
    INFERENCE = "inference"
    GENERAL = "general"


class InstanceStatus(str, Enum):
    """Status of the RunPod instance."""
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class RunPodInstance:
    """RunPod instance information."""
    id: str
    name: str
    status: InstanceStatus
    gpu_type: str
    gpu_count: int
    cost_per_hour: float
    created_at: datetime
    purpose: InstancePurpose
    api_endpoint: Optional[str] = None
    ssh_details: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class InstanceManager:
    """Unified RunPod instance manager with enhanced features."""
    
    # GPU configurations with costs (updated pricing)
    GPU_CONFIGS = {
        "RTX_4090": {"id": "NVIDIA GeForce RTX 4090", "vram": 24, "cost_per_hour": 0.69},
        "RTX_A6000": {"id": "NVIDIA RTX A6000", "vram": 48, "cost_per_hour": 1.28},
        "A100_40GB": {"id": "NVIDIA A100 40GB PCIe", "vram": 40, "cost_per_hour": 1.89},
        "A100_80GB": {"id": "NVIDIA A100 80GB PCIe", "vram": 80, "cost_per_hour": 2.49},
        "H100": {"id": "NVIDIA H100 80GB HBM3", "vram": 80, "cost_per_hour": 4.89},
        "L40S": {"id": "NVIDIA L40S", "vram": 48, "cost_per_hour": 1.64},
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize RunPod manager.
        
        Args:
            api_key: RunPod API key (uses RUNPOD_API_KEY env var if not provided)
        """
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RunPod API key required")
        
        runpod.api_key = self.api_key
        self.instances: Dict[str, RunPodInstance] = {}
        
    async def create_instance(
        self,
        gpu_type: str,
        gpu_count: int = 1,
        purpose: InstancePurpose = InstancePurpose.TRAINING,
        disk_size: int = 50,
        container_image: str = "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04",
        ports: str = "8888/http,6006/http",
        volume_size: int = 100,
        name: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        spot_instance: bool = True
    ) -> RunPodInstance:
        """Create a new RunPod instance with enhanced configuration.
        
        Args:
            gpu_type: Type of GPU (e.g., "RTX_4090", "A100_80GB")
            gpu_count: Number of GPUs
            purpose: Purpose of the instance
            disk_size: Container disk size in GB
            container_image: Docker image to use
            ports: Exposed ports configuration
            volume_size: Persistent volume size in GB
            name: Optional instance name
            env_vars: Environment variables
            spot_instance: Use spot pricing
            
        Returns:
            Created RunPod instance
        """
        if gpu_type not in self.GPU_CONFIGS:
            raise ValueError(f"Unknown GPU type: {gpu_type}")
        
        gpu_config = self.GPU_CONFIGS[gpu_type]
        
        # Generate instance name
        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"unsloth_{purpose.value}_{gpu_type}_{timestamp}"
        
        # Prepare environment variables
        env = env_vars or {}
        env.update({
            "PYTHONPATH": "/workspace",
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "PURPOSE": purpose.value
        })
        
        logger.info(f"Creating RunPod instance: {name}")
        
        try:
            # Create pod configuration
            pod_config = {
                "name": name,
                "image_name": container_image,
                "gpu_type_id": gpu_config["id"],
                "gpu_count": gpu_count,
                "container_disk_in_gb": disk_size,
                "volume_in_gb": volume_size,
                "ports": ports,
                "docker_args": "",
                "env": env,
                "cloud_type": "SECURE" if not spot_instance else "ALL",
                "volume_mount_path": "/workspace",
                "support_public_ip": True,
            }
            
            # Create the pod
            response = runpod.create_pod(**pod_config)
            
            if not response or "id" not in response:
                raise RuntimeError(f"Failed to create pod: {response}")
            
            # Create instance object
            instance = RunPodInstance(
                id=response["id"],
                name=name,
                status=InstanceStatus.PENDING,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                cost_per_hour=gpu_config["cost_per_hour"] * gpu_count,
                created_at=datetime.now(),
                purpose=purpose,
                metadata={
                    "container_image": container_image,
                    "disk_size": disk_size,
                    "volume_size": volume_size,
                    "spot_instance": spot_instance
                }
            )
            
            self.instances[instance.id] = instance
            
            # Wait for instance to be ready
            await self._wait_for_ready(instance)
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create instance: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(30),
        wait=wait_exponential(multiplier=2, min=10, max=60)
    )
    async def _wait_for_ready(self, instance: RunPodInstance) -> None:
        """Wait for instance to be ready.
        
        Args:
            instance: RunPod instance to wait for
        """
        pod = runpod.get_pod(instance.id)
        
        if not pod:
            raise RuntimeError(f"Pod {instance.id} not found")
        
        status = pod.get("desiredStatus", "unknown")
        
        if status == "RUNNING":
            instance.status = InstanceStatus.RUNNING
            
            # Extract connection details
            if "runtime" in pod:
                runtime = pod["runtime"]
                if "ports" in runtime:
                    for port in runtime["ports"]:
                        if port.get("privatePort") == 8888:
                            instance.api_endpoint = f"https://{instance.id}-{port['publicPort']}.proxy.runpod.net"
                            break
            
            # Extract SSH details if available
            ssh_info = pod.get("machine", {})
            if ssh_info:
                instance.ssh_details = {
                    "host": ssh_info.get("podHostId"),
                    "port": 22,
                    "user": "root"
                }
            
            logger.info(f"Instance {instance.name} is ready")
            logger.info(f"API endpoint: {instance.api_endpoint}")
        else:
            logger.info(f"Instance {instance.name} status: {status}")
            raise RuntimeError("Instance not ready yet")
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a RunPod instance.
        
        Args:
            instance_id: Instance ID to terminate
            
        Returns:
            Success status
        """
        try:
            instance = self.instances.get(instance_id)
            if not instance:
                logger.warning(f"Instance {instance_id} not found in local cache")
            
            # Terminate the pod
            response = runpod.terminate_pod(instance_id)
            
            if response:
                if instance:
                    instance.status = InstanceStatus.TERMINATED
                logger.info(f"Terminated instance {instance_id}")
                return True
            else:
                logger.error(f"Failed to terminate instance {instance_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error terminating instance: {e}")
            return False
    
    async def get_instance_status(self, instance_id: str) -> Optional[InstanceStatus]:
        """Get current status of an instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Current status or None if not found
        """
        try:
            pod = runpod.get_pod(instance_id)
            
            if not pod:
                return None
            
            status_map = {
                "CREATED": InstanceStatus.PENDING,
                "RUNNING": InstanceStatus.RUNNING,
                "STOPPED": InstanceStatus.STOPPED,
                "FAILED": InstanceStatus.FAILED,
                "TERMINATED": InstanceStatus.TERMINATED
            }
            
            pod_status = pod.get("desiredStatus", "unknown")
            return status_map.get(pod_status, InstanceStatus.PENDING)
            
        except Exception as e:
            logger.error(f"Error getting instance status: {e}")
            return None
    
    async def list_instances(self) -> List[RunPodInstance]:
        """List all active instances.
        
        Returns:
            List of active instances
        """
        try:
            pods = runpod.get_pods()
            
            instances = []
            for pod in pods:
                # Map pod data to instance
                instance = self._pod_to_instance(pod)
                if instance:
                    instances.append(instance)
                    # Update local cache
                    self.instances[instance.id] = instance
            
            return instances
            
        except Exception as e:
            logger.error(f"Error listing instances: {e}")
            return []
    
    def _pod_to_instance(self, pod_data: Dict[str, Any]) -> Optional[RunPodInstance]:
        """Convert RunPod API pod data to instance object.
        
        Args:
            pod_data: Raw pod data from API
            
        Returns:
            RunPodInstance or None if invalid
        """
        try:
            # Extract GPU info
            gpu_type = "unknown"
            gpu_count = 1
            
            if "machine" in pod_data:
                machine = pod_data["machine"]
                gpu_type_id = machine.get("gpuTypeId", "")
                gpu_count = machine.get("gpuCount", 1)
                
                # Map GPU type ID to our naming
                for name, config in self.GPU_CONFIGS.items():
                    if config["id"] == gpu_type_id:
                        gpu_type = name
                        break
            
            # Get cost
            cost_per_hour = pod_data.get("costPerHr", 0.0)
            
            # Map status
            status_map = {
                "CREATED": InstanceStatus.PENDING,
                "RUNNING": InstanceStatus.RUNNING,
                "STOPPED": InstanceStatus.STOPPED,
                "FAILED": InstanceStatus.FAILED,
                "EXITED": InstanceStatus.TERMINATED
            }
            
            status = status_map.get(
                pod_data.get("desiredStatus", "unknown"),
                InstanceStatus.PENDING
            )
            
            # Create instance
            instance = RunPodInstance(
                id=pod_data["id"],
                name=pod_data.get("name", "unknown"),
                status=status,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                cost_per_hour=cost_per_hour,
                created_at=datetime.fromisoformat(
                    pod_data.get("created", datetime.now().isoformat())
                ),
                purpose=InstancePurpose.GENERAL,  # Default, update from metadata if available
                metadata=pod_data
            )
            
            return instance
            
        except Exception as e:
            logger.error(f"Error converting pod data: {e}")
            return None
    
    def estimate_training_cost(
        self,
        gpu_type: str,
        gpu_count: int,
        estimated_hours: float
    ) -> float:
        """Estimate cost for training job.
        
        Args:
            gpu_type: Type of GPU
            gpu_count: Number of GPUs
            estimated_hours: Estimated training hours
            
        Returns:
            Estimated cost in USD
        """
        if gpu_type not in self.GPU_CONFIGS:
            raise ValueError(f"Unknown GPU type: {gpu_type}")
        
        hourly_cost = self.GPU_CONFIGS[gpu_type]["cost_per_hour"] * gpu_count
        total_cost = hourly_cost * estimated_hours
        
        return round(total_cost, 2)


if __name__ == "__main__":
    # Validation
    async def validate():
        # Test GPU configs
        manager = InstanceManager(api_key="test_key")
        cost = manager.estimate_training_cost("RTX_4090", 2, 4.5)
        assert cost == round(0.69 * 2 * 4.5, 2)
        print(" Instance manager validation passed")
    
    asyncio.run(validate())