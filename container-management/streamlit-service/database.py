"""
SQLite database operations for container deployment tracking.
"""
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import contextmanager

from models import User, Deployment, DeploymentStatus, HealthCheck


# Database schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    organization TEXT,
    organizational_unit TEXT,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE IF NOT EXISTS deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    container_name TEXT NOT NULL,
    container_id TEXT,
    vm_name TEXT NOT NULL,
    vm_host TEXT NOT NULL,
    project_path TEXT,
    image_name TEXT NOT NULL,
    image_tag TEXT DEFAULT 'latest',
    ports TEXT,
    memory_limit TEXT DEFAULT '32g',
    gpu_enabled BOOLEAN DEFAULT TRUE,
    status TEXT DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_health_check TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    error_message TEXT,
    FOREIGN KEY (deployment_id) REFERENCES deployments(id)
);

CREATE INDEX IF NOT EXISTS idx_deployments_user ON deployments(user_id);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
CREATE INDEX IF NOT EXISTS idx_deployments_vm ON deployments(vm_name);
CREATE INDEX IF NOT EXISTS idx_deployments_container ON deployments(container_name);
"""

# Default database path
DEFAULT_DB_PATH = os.path.join(
    os.path.expanduser("~"),
    ".container-deployments.db"
)


class Database:
    """SQLite database manager for deployment tracking."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("DB_PATH", DEFAULT_DB_PATH)
        self._init_db()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA)

    # -------------------------------------------------------------------------
    # User Operations
    # -------------------------------------------------------------------------

    def get_or_create_user(self, user: User) -> User:
        """Get existing user or create new one."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (user.username,)
            )
            row = cursor.fetchone()

            if row:
                user.id = row["id"]
                user.is_admin = bool(row["is_admin"])
                user.created_at = row["created_at"]
                # Update last_login
                conn.execute(
                    "UPDATE users SET last_login = ? WHERE id = ?",
                    (datetime.now().isoformat(), user.id)
                )
            else:
                cursor = conn.execute(
                    """INSERT INTO users
                       (username, email, organization, organizational_unit, is_admin, last_login)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (user.username, user.email, user.organization,
                     user.organizational_unit, user.is_admin, datetime.now().isoformat())
                )
                user.id = cursor.lastrowid

            return user

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None

    def get_all_users(self) -> List[User]:
        """Get all users (for admin view)."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users ORDER BY username")
            return [self._row_to_user(row) for row in cursor.fetchall()]

    def update_user_admin_status(self, user_id: int, is_admin: bool):
        """Update user admin status."""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE users SET is_admin = ? WHERE id = ?",
                (is_admin, user_id)
            )

    # -------------------------------------------------------------------------
    # Deployment Operations
    # -------------------------------------------------------------------------

    def create_deployment(self, deployment: Deployment) -> Deployment:
        """Create a new deployment record."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO deployments
                   (user_id, container_name, container_id, vm_name, vm_host,
                    project_path, image_name, image_tag, ports, memory_limit,
                    gpu_enabled, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (deployment.user_id, deployment.container_name,
                 deployment.container_id, deployment.vm_name, deployment.vm_host,
                 deployment.project_path, deployment.image_name, deployment.image_tag,
                 json.dumps(deployment.ports), deployment.memory_limit,
                 deployment.gpu_enabled, deployment.status.value)
            )
            deployment.id = cursor.lastrowid
            return deployment

    def get_deployment_by_id(self, deployment_id: int) -> Optional[Deployment]:
        """Get deployment by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT d.*, u.username
                   FROM deployments d
                   JOIN users u ON d.user_id = u.id
                   WHERE d.id = ?""",
                (deployment_id,)
            )
            row = cursor.fetchone()
            return self._row_to_deployment(row) if row else None

    def get_deployment_by_container(self, container_name: str, vm_name: str = None) -> Optional[Deployment]:
        """Get deployment by container name and optionally VM."""
        with self.get_connection() as conn:
            if vm_name:
                cursor = conn.execute(
                    """SELECT d.*, u.username
                       FROM deployments d
                       JOIN users u ON d.user_id = u.id
                       WHERE d.container_name = ? AND d.vm_name = ?""",
                    (container_name, vm_name)
                )
            else:
                cursor = conn.execute(
                    """SELECT d.*, u.username
                       FROM deployments d
                       JOIN users u ON d.user_id = u.id
                       WHERE d.container_name = ?
                       ORDER BY d.created_at DESC LIMIT 1""",
                    (container_name,)
                )
            row = cursor.fetchone()
            return self._row_to_deployment(row) if row else None

    def get_user_deployments(self, user_id: int, active_only: bool = True) -> List[Deployment]:
        """Get all deployments for a user."""
        with self.get_connection() as conn:
            query = """SELECT d.*, u.username
                       FROM deployments d
                       JOIN users u ON d.user_id = u.id
                       WHERE d.user_id = ?"""
            if active_only:
                query += " AND d.status NOT IN ('removed')"
            query += " ORDER BY d.created_at DESC"

            cursor = conn.execute(query, (user_id,))
            return [self._row_to_deployment(row) for row in cursor.fetchall()]

    def get_all_deployments(self, filters: Dict = None) -> List[Deployment]:
        """Get all deployments with optional filters (for admin)."""
        with self.get_connection() as conn:
            query = """SELECT d.*, u.username
                       FROM deployments d
                       JOIN users u ON d.user_id = u.id
                       WHERE 1=1"""
            params = []

            if filters:
                if filters.get("user_id"):
                    query += " AND d.user_id = ?"
                    params.append(filters["user_id"])
                if filters.get("vm_name"):
                    query += " AND d.vm_name = ?"
                    params.append(filters["vm_name"])
                if filters.get("status"):
                    query += " AND d.status = ?"
                    params.append(filters["status"])
                if not filters.get("include_removed", False):
                    query += " AND d.status != 'removed'"

            query += " ORDER BY d.created_at DESC"
            cursor = conn.execute(query, params)
            return [self._row_to_deployment(row) for row in cursor.fetchall()]

    def update_deployment_status(self, deployment_id: int, status: DeploymentStatus,
                                  container_id: Optional[str] = None):
        """Update deployment status."""
        with self.get_connection() as conn:
            if container_id:
                conn.execute(
                    """UPDATE deployments
                       SET status = ?, container_id = ?, updated_at = ?
                       WHERE id = ?""",
                    (status.value, container_id, datetime.now().isoformat(), deployment_id)
                )
            else:
                conn.execute(
                    "UPDATE deployments SET status = ?, updated_at = ? WHERE id = ?",
                    (status.value, datetime.now().isoformat(), deployment_id)
                )

    def update_health_check(self, deployment_id: int, status: str):
        """Update last health check timestamp and status."""
        with self.get_connection() as conn:
            conn.execute(
                """UPDATE deployments
                   SET last_health_check = ?, status = ?, updated_at = ?
                   WHERE id = ?""",
                (datetime.now().isoformat(), status, datetime.now().isoformat(), deployment_id)
            )

    # -------------------------------------------------------------------------
    # Health Check Operations
    # -------------------------------------------------------------------------

    def log_health_check(self, check: HealthCheck):
        """Log a health check result."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO health_checks
                   (deployment_id, status, response_time_ms, error_message)
                   VALUES (?, ?, ?, ?)""",
                (check.deployment_id, check.status,
                 check.response_time_ms, check.error_message)
            )

    def get_recent_health_checks(self, deployment_id: int, limit: int = 10) -> List[HealthCheck]:
        """Get recent health checks for a deployment."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT * FROM health_checks
                   WHERE deployment_id = ?
                   ORDER BY checked_at DESC LIMIT ?""",
                (deployment_id, limit)
            )
            return [self._row_to_health_check(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Statistics (for admin dashboard)
    # -------------------------------------------------------------------------

    def get_deployment_stats(self) -> Dict:
        """Get deployment statistics for admin dashboard."""
        with self.get_connection() as conn:
            stats = {}

            # Total deployments by status
            cursor = conn.execute(
                """SELECT status, COUNT(*) as count
                   FROM deployments GROUP BY status"""
            )
            stats["by_status"] = {row["status"]: row["count"] for row in cursor}

            # Deployments by VM
            cursor = conn.execute(
                """SELECT vm_name, COUNT(*) as count
                   FROM deployments WHERE status = 'running'
                   GROUP BY vm_name"""
            )
            stats["by_vm"] = {row["vm_name"]: row["count"] for row in cursor}

            # Deployments by user (top 10)
            cursor = conn.execute(
                """SELECT u.username, COUNT(*) as count
                   FROM deployments d
                   JOIN users u ON d.user_id = u.id
                   WHERE d.status = 'running'
                   GROUP BY d.user_id
                   ORDER BY count DESC
                   LIMIT 10"""
            )
            stats["by_user"] = {row["username"]: row["count"] for row in cursor}

            # Total users
            cursor = conn.execute("SELECT COUNT(*) as count FROM users")
            stats["total_users"] = cursor.fetchone()["count"]

            # Total deployments
            cursor = conn.execute("SELECT COUNT(*) as count FROM deployments WHERE status != 'removed'")
            stats["total_deployments"] = cursor.fetchone()["count"]

            return stats

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _row_to_user(self, row) -> User:
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            organization=row["organization"],
            organizational_unit=row["organizational_unit"],
            is_admin=bool(row["is_admin"]),
            created_at=row["created_at"],
            last_login=row["last_login"],
        )

    def _row_to_deployment(self, row) -> Deployment:
        status_value = row["status"]
        try:
            status = DeploymentStatus(status_value)
        except ValueError:
            status = DeploymentStatus.UNKNOWN

        return Deployment(
            id=row["id"],
            user_id=row["user_id"],
            container_name=row["container_name"],
            container_id=row["container_id"],
            vm_name=row["vm_name"],
            vm_host=row["vm_host"],
            project_path=row["project_path"],
            image_name=row["image_name"],
            image_tag=row["image_tag"],
            ports=json.loads(row["ports"]) if row["ports"] else {},
            memory_limit=row["memory_limit"],
            gpu_enabled=bool(row["gpu_enabled"]),
            status=status,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_health_check=row["last_health_check"],
            username=row["username"] if "username" in row.keys() else None,
        )

    def _row_to_health_check(self, row) -> HealthCheck:
        return HealthCheck(
            id=row["id"],
            deployment_id=row["deployment_id"],
            status=row["status"],
            checked_at=row["checked_at"],
            response_time_ms=row["response_time_ms"],
            error_message=row["error_message"],
        )


# Singleton instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get or create database singleton."""
    global _db
    if _db is None:
        _db = Database()
    return _db
