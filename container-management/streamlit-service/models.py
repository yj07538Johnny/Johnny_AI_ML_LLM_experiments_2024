"""
Data models for the container deployment service.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict
from enum import Enum


class DeploymentStatus(Enum):
    """Status of a container deployment."""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"
    REMOVED = "removed"


@dataclass
class User:
    """User model for database storage."""
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    organization: str = ""
    organizational_unit: str = ""
    is_admin: bool = False
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


@dataclass
class Deployment:
    """Deployment record model."""
    id: Optional[int] = None
    user_id: int = 0
    container_name: str = ""
    container_id: Optional[str] = None
    vm_name: str = ""
    vm_host: str = ""
    project_path: Optional[str] = None
    image_name: str = ""
    image_tag: str = "latest"
    ports: Dict[str, str] = field(default_factory=dict)
    memory_limit: str = "32g"
    gpu_enabled: bool = True
    status: DeploymentStatus = DeploymentStatus.UNKNOWN
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    # Joined field (populated on query, not stored)
    username: Optional[str] = None


@dataclass
class HealthCheck:
    """Health check record."""
    id: Optional[int] = None
    deployment_id: int = 0
    status: str = ""
    checked_at: Optional[datetime] = None
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
