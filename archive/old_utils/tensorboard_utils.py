import os
from pathlib import Path
import subprocess
import sys
import time
import webbrowser

from loguru import logger

from app.backend.unsloth_wip.phi3_5_training import TrainingConfig


def verify_tensorboard() -> bool:
    """Verify TensorBoard installation and availability."""
    try:
        import tensorboard
        logger.info(f"TensorBoard version: {tensorboard.__version__}")
        
        result = subprocess.run(
            ["tensorboard", "--version"], 
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"TensorBoard CLI version: {result.stdout.strip()}")
            return True
    except ImportError:
        logger.error("TensorBoard not installed. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
            logger.success("TensorBoard installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install TensorBoard: {e}")
    except Exception as e:
        logger.error(f"Error verifying TensorBoard: {e}")
    return False

def setup_tensorboard(config: TrainingConfig) -> str:
    """Setup and launch TensorBoard server using Docker."""
    log_dir = Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if os.name == 'nt':
        try:
            wsl_path = subprocess.run(
                ["wsl", "wslpath", "-a", str(log_dir.absolute())],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            mount_path = wsl_path
        except Exception:
            logger.warning("Failed to convert Windows path, using absolute path")
            mount_path = str(log_dir.absolute())
    else:
        mount_path = str(log_dir.absolute())
    
    try:
        subprocess.run(
            ["docker", "rm", "-f", "training-tensorboard"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
    except Exception as e:
        logger.warning(f"Failed to kill existing TensorBoard container: {e}")
    
    try:
        cmd = [
            "docker", "run",
            "--name", "training-tensorboard",
            "-d",
            "-p", f"{config.tensorboard_port}:6006",
            "-v", f"{mount_path}:/logs",
            "--restart", "unless-stopped",
            "--add-host=host.docker.internal:host-gateway",
            "-e", "PYTHONUNBUFFERED=1",
            "tensorflow/tensorflow:latest",
            "tensorboard",
            "--logdir", "/logs",
            "--host", "0.0.0.0",
            "--reload_interval", "5"
        ]
        
        logger.debug(f"Docker command: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        container_id = process.stdout.strip()
        logger.info(f"TensorBoard container started: {container_id}")
        
        time.sleep(3)
        
        status = subprocess.run(
            ["docker", "ps", "--filter", f"id={container_id}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if not status.stdout.strip():
            logs = subprocess.run(
                ["docker", "logs", container_id],
                capture_output=True,
                text=True,
                check=True
            )
            raise RuntimeError(f"TensorBoard container failed to start. Logs:\n{logs.stdout}\n{logs.stderr}")
            
        logger.info(f"TensorBoard available at http://localhost:{config.tensorboard_port}")
        
        if config.auto_launch_browser:
            url = f"http://localhost:{config.tensorboard_port}"
            logger.info(f"Opening TensorBoard at {url}")
            webbrowser.open(url)
        
        return container_id
        
    except Exception as e:
        logger.error(f"Failed to start TensorBoard container: {e}")
        raise

def cleanup_tensorboard(container_id: str):
    """Cleanup TensorBoard Docker container."""
    if container_id:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                check=True,
                capture_output=True
            )
            logger.info("TensorBoard container cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup TensorBoard container: {e}")
