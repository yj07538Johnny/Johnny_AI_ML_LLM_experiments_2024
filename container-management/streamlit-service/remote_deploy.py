"""
Remote Deployment Module

Provides SSH-based deployment functions for deploying containers to remote VMs.
Supports both local (direct docker) and remote (SSH + docker) deployment patterns.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import yaml

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False


@dataclass
class VMConfig:
    """Configuration for a target VM."""
    name: str
    host: str
    ssh_port: Optional[int]
    ssh_user: Optional[str]
    root_directory: str
    description: str = ""

    @property
    def is_local(self) -> bool:
        """Check if this is a local deployment (no SSH needed)."""
        return self.ssh_port is None or self.host == "localhost"


def load_vm_config(config_path: Optional[str] = None) -> List[VMConfig]:
    """
    Load VM configuration from YAML file.

    Args:
        config_path: Path to vm_config.yaml. If None, uses default location.

    Returns:
        List of VMConfig objects
    """
    if config_path is None:
        config_path = Path(__file__).parent / "vm_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    vms = []
    for vm_data in config.get('vms', []):
        vms.append(VMConfig(
            name=vm_data.get('name', 'Unknown'),
            host=vm_data.get('host', 'localhost'),
            ssh_port=vm_data.get('ssh_port'),
            ssh_user=vm_data.get('ssh_user'),
            root_directory=vm_data.get('root_directory', '/tmp'),
            description=vm_data.get('description', ''),
        ))

    return vms


def sanitize_project_name(name: str) -> str:
    """
    Sanitize project name for use as directory name.

    - Replaces spaces with underscores
    - Removes special characters except underscores and hyphens
    - Converts to lowercase

    Args:
        name: Raw project name from user input

    Returns:
        Sanitized name safe for filesystem use
    """
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove special characters except alphanumeric, underscore, hyphen
    name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    # Convert to lowercase for consistency
    name = name.lower()
    # Remove leading/trailing underscores
    name = name.strip('_-')
    # Default name if empty
    if not name:
        name = "unnamed_project"
    return name


def get_ssh_client(vm: VMConfig) -> 'paramiko.SSHClient':
    """
    Create and return an SSH client connected to the VM.

    Args:
        vm: VM configuration

    Returns:
        Connected paramiko SSHClient

    Raises:
        ImportError: If paramiko is not installed
        Exception: If connection fails
    """
    if not PARAMIKO_AVAILABLE:
        raise ImportError("paramiko is required for SSH connections. Install with: pip install paramiko")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Try to use SSH agent or default key locations
    client.connect(
        hostname=vm.host,
        port=vm.ssh_port or 22,
        username=vm.ssh_user,
        allow_agent=True,
        look_for_keys=True,
    )

    return client


def test_vm_connection(vm: VMConfig) -> Tuple[bool, str]:
    """
    Test SSH connectivity to a VM.

    Args:
        vm: VM configuration to test

    Returns:
        Tuple of (success: bool, message: str)
    """
    if vm.is_local:
        # Test local docker
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return True, "Local Docker is available"
            else:
                return False, f"Docker error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "Docker command timed out"
        except FileNotFoundError:
            return False, "Docker not found"

    # Test SSH connection
    try:
        client = get_ssh_client(vm)
        stdin, stdout, stderr = client.exec_command("docker info")
        exit_code = stdout.channel.recv_exit_status()
        client.close()

        if exit_code == 0:
            return True, f"Connected to {vm.host} - Docker available"
        else:
            return False, f"Docker not available on {vm.host}"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"SSH connection failed: {str(e)}"


def create_project_directory(
    vm: VMConfig,
    project_name: str,
    status_callback=None
) -> Tuple[bool, str, str]:
    """
    Create project directory structure on target VM.

    Creates:
        {root_directory}/{project_name}/
        ├── workspace/  (mounted to /workspace)
        └── data/       (mounted to /data, read-only)

    Args:
        vm: Target VM configuration
        project_name: Sanitized project name
        status_callback: Optional callback function for status updates

    Returns:
        Tuple of (success: bool, message: str, project_path: str)
    """
    project_path = f"{vm.root_directory}/{project_name}"

    commands = [
        f"mkdir -p {project_path}/workspace",
        f"mkdir -p {project_path}/data",
    ]

    def update_status(msg):
        if status_callback:
            status_callback(msg)

    if vm.is_local:
        # Local execution
        try:
            for cmd in commands:
                update_status(f"Running: {cmd}")
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    return False, f"Failed to create directory: {result.stderr}", ""

            return True, f"Created {project_path}", project_path
        except Exception as e:
            return False, f"Error: {str(e)}", ""

    # Remote execution via SSH
    try:
        update_status(f"Connecting to {vm.host}...")
        client = get_ssh_client(vm)

        for cmd in commands:
            update_status(f"Running: {cmd}")
            stdin, stdout, stderr = client.exec_command(cmd)
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                error = stderr.read().decode()
                client.close()
                return False, f"Failed: {error}", ""

        client.close()
        return True, f"Created {project_path} on {vm.host}", project_path

    except ImportError as e:
        return False, str(e), ""
    except Exception as e:
        return False, f"SSH error: {str(e)}", ""


def deploy_container_remote(
    vm: VMConfig,
    project_path: str,
    image: str,
    container_name: str,
    ports: Dict[str, str],
    gpu: bool = True,
    memory_limit: str = "32g",
    user_id: str = "",
    user_email: str = "",
    status_callback=None
) -> Dict:
    """
    Deploy a container on the target VM.

    Args:
        vm: Target VM configuration
        project_path: Full path to project directory
        image: Docker image to deploy
        container_name: Name for the container
        ports: Dict mapping host_port -> container_port
        gpu: Whether to enable GPU access
        memory_limit: Memory limit (e.g., "32g")
        user_id: User ID for environment variable
        user_email: User email for environment variable
        status_callback: Optional callback for status updates

    Returns:
        Dict with success, error, container_id, access_urls
    """
    def update_status(msg):
        if status_callback:
            status_callback(msg)

    # Build docker run command
    cmd_parts = [
        "docker", "run", "-d",
        "--name", container_name,
        "--memory", memory_limit,
        "--shm-size", "2g",
    ]

    # Add GPU support
    if gpu:
        cmd_parts.extend(["--gpus", "all"])

    # Add environment variables
    if user_id:
        cmd_parts.extend(["-e", f"USER_ID={user_id}"])
    if user_email:
        cmd_parts.extend(["-e", f"USER_EMAIL={user_email}"])

    # Add volume mounts
    cmd_parts.extend(["-v", f"{project_path}/workspace:/workspace"])
    cmd_parts.extend(["-v", f"{project_path}/data:/data:ro"])

    # Add port mappings
    port_mappings = {}
    for host_port, container_port in ports.items():
        cmd_parts.extend(["-p", f"{host_port}:{container_port}"])
        port_mappings[container_port] = host_port

    # Add image
    cmd_parts.append(image)

    # Build the command string for SSH
    cmd_str = " ".join(cmd_parts)

    # First, stop and remove existing container if it exists
    stop_cmd = f"docker stop {container_name} 2>/dev/null; docker rm {container_name} 2>/dev/null; true"

    if vm.is_local:
        try:
            # Stop existing
            update_status("Stopping existing container if any...")
            subprocess.run(stop_cmd, shell=True, capture_output=True, timeout=30)

            # Deploy new
            update_status(f"Deploying {container_name}...")
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr,
                }

            container_id = result.stdout.strip()[:12]

            # Build access URLs
            access_urls = {}
            for container_port, host_port in port_mappings.items():
                access_urls[container_port] = f"http://{vm.host}:{host_port}"

            return {
                "success": True,
                "container_id": container_id,
                "container_name": container_name,
                "access_urls": access_urls,
                "host": vm.host,
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Deployment timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Remote deployment via SSH
    try:
        update_status(f"Connecting to {vm.host}...")
        client = get_ssh_client(vm)

        # Stop existing
        update_status("Stopping existing container if any...")
        stdin, stdout, stderr = client.exec_command(stop_cmd)
        stdout.channel.recv_exit_status()

        # Deploy new
        update_status(f"Deploying {container_name}...")
        stdin, stdout, stderr = client.exec_command(cmd_str)
        exit_code = stdout.channel.recv_exit_status()

        if exit_code != 0:
            error = stderr.read().decode()
            client.close()
            return {"success": False, "error": error}

        container_id = stdout.read().decode().strip()[:12]
        client.close()

        # Build access URLs
        access_urls = {}
        for container_port, host_port in port_mappings.items():
            access_urls[container_port] = f"http://{vm.host}:{host_port}"

        return {
            "success": True,
            "container_id": container_id,
            "container_name": container_name,
            "access_urls": access_urls,
            "host": vm.host,
        }

    except ImportError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"SSH error: {str(e)}"}


def get_container_status_remote(
    vm: VMConfig,
    container_name: str
) -> Dict:
    """
    Get status of a container on the target VM.

    Args:
        vm: Target VM configuration
        container_name: Name of the container

    Returns:
        Dict with status, running, ports, etc.
    """
    cmd = f"docker inspect --format '{{{{.State.Status}}}} {{{{.State.Running}}}}' {container_name}"

    if vm.is_local:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}} {{.State.Running}}", container_name],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {"exists": False, "running": False}

            parts = result.stdout.strip().split()
            return {
                "exists": True,
                "status": parts[0] if parts else "unknown",
                "running": parts[1].lower() == "true" if len(parts) > 1 else False,
            }
        except Exception:
            return {"exists": False, "running": False}

    # Remote via SSH
    try:
        client = get_ssh_client(vm)
        stdin, stdout, stderr = client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()

        if exit_code != 0:
            client.close()
            return {"exists": False, "running": False}

        output = stdout.read().decode().strip()
        parts = output.split()
        client.close()

        return {
            "exists": True,
            "status": parts[0] if parts else "unknown",
            "running": parts[1].lower() == "true" if len(parts) > 1 else False,
        }
    except Exception:
        return {"exists": False, "running": False}


def stop_container_remote(vm: VMConfig, container_name: str) -> Tuple[bool, str]:
    """
    Stop a container on the target VM.

    Args:
        vm: Target VM configuration
        container_name: Name of the container to stop

    Returns:
        Tuple of (success: bool, message: str)
    """
    cmd = f"docker stop {container_name}"

    if vm.is_local:
        try:
            result = subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return True, f"Stopped {container_name}"
            else:
                return False, result.stderr
        except Exception as e:
            return False, str(e)

    # Remote via SSH
    try:
        client = get_ssh_client(vm)
        stdin, stdout, stderr = client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        client.close()

        if exit_code == 0:
            return True, f"Stopped {container_name} on {vm.host}"
        else:
            return False, stderr.read().decode()
    except Exception as e:
        return False, str(e)


def list_containers_remote(vm: VMConfig, user_filter: Optional[str] = None) -> List[Dict]:
    """
    List containers on the target VM.

    Args:
        vm: Target VM configuration
        user_filter: Optional user name to filter by (container name suffix)

    Returns:
        List of container dicts with name, image, status, ports
    """
    cmd = 'docker ps --format \'{"name":"{{.Names}}","image":"{{.Image}}","status":"{{.Status}}","ports":"{{.Ports}}"}\''

    def parse_output(output: str) -> List[Dict]:
        import json
        containers = []
        for line in output.strip().split('\n'):
            if line:
                try:
                    container = json.loads(line)
                    if user_filter is None or container["name"].endswith(f"-{user_filter}"):
                        containers.append(container)
                except json.JSONDecodeError:
                    continue
        return containers

    if vm.is_local:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            return parse_output(result.stdout)
        except Exception:
            return []

    # Remote via SSH
    try:
        client = get_ssh_client(vm)
        stdin, stdout, stderr = client.exec_command(cmd)
        stdout.channel.recv_exit_status()
        output = stdout.read().decode()
        client.close()
        return parse_output(output)
    except Exception:
        return []
