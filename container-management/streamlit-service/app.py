"""
Container Deployment Service

A Streamlit-based web interface for deploying curated GPU containers
to data scientist VMs with PKI authentication.

Prerequisites:
- Streamlit
- Docker SDK for Python
- PKI certificates configured on web server (Apache/nginx)

The web server handles PKI validation and passes the validated
certificate info to Streamlit via headers.
"""

import os
import json
import subprocess
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system env vars

import streamlit as st

from remote_deploy import (
    load_vm_config, deploy_container_remote,
    create_project_directory, sanitize_project_name,
    get_container_status_remote, stop_container_remote,
    list_containers_remote, test_vm_connection, VMConfig
)

# Import database and models
from database import get_db
from models import User, Deployment, DeploymentStatus

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "registry": "ghcr.io/your-org",
    "curated_images": [
        {
            "name": "gpu-jupyter",
            "description": "Jupyter Lab with PyTorch, TensorFlow, GPU support",
            "tag": "latest",
            "ports": {"8888": "Jupyter Lab", "6006": "TensorBoard"},
            "gpu": True,
        },
        {
            "name": "gpu-pytorch",
            "description": "PyTorch environment for training",
            "tag": "latest",
            "ports": {"8888": "Jupyter Lab"},
            "gpu": True,
        },
        {
            "name": "ml-cpu",
            "description": "CPU-only ML environment (lighter weight)",
            "tag": "latest",
            "ports": {"8888": "Jupyter Lab"},
            "gpu": False,
        },
    ],
    "user_port_start": 10000,
    "vm_host": os.getenv("VM_HOST", "localhost"),
    "log_file": os.path.join(os.path.expanduser("~"), ".container-deployments.log"),
    "admin_users": [u.strip() for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip()],
}

# =============================================================================
# PKI AUTHENTICATION
# =============================================================================

@dataclass
class UserIdentity:
    """User identity extracted from PKI certificate."""
    common_name: str
    email: str
    organization: str
    organizational_unit: str
    is_admin: bool = False


def get_user_from_pki() -> Optional[UserIdentity]:
    """Extract user identity from PKI certificate headers."""
    # In development, use environment variable override
    if os.getenv("DEV_USER"):
        username = os.getenv("DEV_USER")
        return UserIdentity(
            common_name=username,
            email=f"{username}@dev.local",
            organization="Development",
            organizational_unit="DataScience",
            is_admin=os.getenv("DEV_ADMIN", "false").lower() == "true" or username in CONFIG["admin_users"],
        )

    # Production: get from headers (set by reverse proxy)
    headers = {}
    try:
        if hasattr(st, 'context') and hasattr(st.context, 'headers'):
            headers = st.context.headers
    except Exception:
        pass

    cn = headers.get("SSL_CLIENT_S_DN_CN") or headers.get("X-SSL-Client-CN")
    if not cn:
        return None

    return UserIdentity(
        common_name=cn,
        email=headers.get("SSL_CLIENT_S_DN_Email", f"{cn}@unknown"),
        organization=headers.get("SSL_CLIENT_S_DN_O", "Unknown"),
        organizational_unit=headers.get("SSL_CLIENT_S_DN_OU", "Unknown"),
        is_admin=cn in CONFIG["admin_users"],
    )


def get_or_select_user() -> Optional[UserIdentity]:
    """
    Get user from PKI or allow simple username selection.
    Priority: PKI > DEV_USER > Session Selection
    """
    # Try PKI first
    user = get_user_from_pki()
    if user:
        return user

    # Check for simple auth mode
    if not os.getenv("ENABLE_SIMPLE_AUTH", "false").lower() == "true":
        return None

    # Simple auth: username selection/entry
    st.sidebar.markdown("### Login")

    # Get existing users from database
    db = get_db()
    existing_users = [u.username for u in db.get_all_users()]

    if existing_users:
        auth_mode = st.sidebar.radio(
            "Login method",
            ["Select existing user", "New user"],
            key="auth_mode",
            horizontal=True
        )
    else:
        auth_mode = "New user"

    if auth_mode == "Select existing user" and existing_users:
        username = st.sidebar.selectbox(
            "Select user",
            options=existing_users,
            key="user_select"
        )
    else:
        username = st.sidebar.text_input(
            "Username",
            key="username_input",
            placeholder="Enter your username"
        )

    if not username:
        return None

    # Check if admin
    is_admin = username in CONFIG["admin_users"]

    return UserIdentity(
        common_name=username,
        email=f"{username}@local",
        organization="Local",
        organizational_unit="General",
        is_admin=is_admin,
    )


def require_auth() -> UserIdentity:
    """Require valid authentication or show error."""
    user = get_or_select_user()
    if not user:
        st.info("Please log in using the sidebar.")
        st.stop()
    return user


# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

def get_user_port_range(user_index: int) -> tuple:
    """Get the port range allocated to a user."""
    start = CONFIG["user_port_start"] + (user_index * 100)
    return (start, start + 99)


def run_docker_command(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run a docker command and return result."""
    return subprocess.run(
        ["docker"] + cmd,
        capture_output=True,
        text=True,
    )


def get_running_containers(user: str = None, curated_only: bool = True) -> List[Dict]:
    """Get list of running containers, optionally filtered by curated images."""
    result = run_docker_command([
        "ps", "--format",
        '{"name":"{{.Names}}","image":"{{.Image}}","status":"{{.Status}}","ports":"{{.Ports}}"}'
    ])

    curated_image_names = [img["name"] for img in CONFIG["curated_images"]]

    containers = []
    for line in result.stdout.strip().split("\n"):
        if line:
            try:
                container = json.loads(line)
                image_name = container["image"].split("/")[-1].split(":")[0]
                is_curated = any(name in container["name"] or name in image_name
                                for name in curated_image_names)
                is_user_container = user and user in container["name"]

                if not curated_only or is_curated or is_user_container:
                    containers.append(container)
            except json.JSONDecodeError:
                continue
    return containers


def stop_container(container_name: str, user: UserIdentity) -> Dict:
    """Stop a user's container."""
    curated_image_names = [img["name"] for img in CONFIG["curated_images"]]
    is_curated = any(name in container_name for name in curated_image_names)

    if not user.is_admin and not is_curated:
        return {"success": False, "error": "Permission denied"}

    result = run_docker_command(["stop", container_name])
    if result.returncode != 0:
        return {"success": False, "error": result.stderr}

    return {"success": True}


def log_deployment(user: UserIdentity, image: str, container: str, ports: Dict):
    """Log deployment for audit trail (legacy JSON log)."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user.common_name,
        "email": user.email,
        "ou": user.organizational_unit,
        "image": image,
        "container": container,
        "ports": ports,
    }

    try:
        with open(CONFIG["log_file"], "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        st.warning(f"Could not write to log: {e}")


# =============================================================================
# ADMIN DASHBOARD
# =============================================================================

def render_admin_dashboard(user_identity: UserIdentity, db_user: User):
    """Render the admin dashboard tab."""
    st.header("Admin Dashboard")

    db = get_db()

    # Statistics Overview
    stats = db.get_deployment_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        running = stats["by_status"].get("running", 0)
        st.metric("Running", running)

    with col2:
        stopped = stats["by_status"].get("stopped", 0)
        st.metric("Stopped", stopped)

    with col3:
        st.metric("Total Users", stats["total_users"])

    with col4:
        st.metric("Total Deployments", stats["total_deployments"])

    st.divider()

    # Filters
    st.subheader("All Deployments")

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        users = db.get_all_users()
        user_options = ["All Users"] + [u.username for u in users]
        selected_user_filter = st.selectbox("Filter by User", user_options, key="admin_user_filter")

    with filter_col2:
        try:
            vms = load_vm_config()
            vm_options = ["All VMs"] + [vm.name for vm in vms]
        except Exception:
            vm_options = ["All VMs"]
        selected_vm_filter = st.selectbox("Filter by VM", vm_options, key="admin_vm_filter")

    with filter_col3:
        status_options = ["All Statuses", "running", "stopped", "error", "unknown"]
        selected_status_filter = st.selectbox("Filter by Status", status_options, key="admin_status_filter")

    with filter_col4:
        include_removed = st.checkbox("Include removed", value=False, key="admin_include_removed")

    # Build filters dict
    filters = {"include_removed": include_removed}
    if selected_user_filter != "All Users":
        user_obj = next((u for u in users if u.username == selected_user_filter), None)
        if user_obj:
            filters["user_id"] = user_obj.id
    if selected_vm_filter != "All VMs":
        filters["vm_name"] = selected_vm_filter
    if selected_status_filter != "All Statuses":
        filters["status"] = selected_status_filter

    # Deployments Table
    deployments = db.get_all_deployments(filters)

    col_refresh, col_health = st.columns([1, 4])
    with col_refresh:
        if st.button("Refresh", key="admin_refresh"):
            st.rerun()

    if not deployments:
        st.info("No deployments found matching the filters.")
    else:
        # Display as table using pandas if available
        try:
            import pandas as pd

            df_data = []
            for dep in deployments:
                df_data.append({
                    "ID": dep.id,
                    "Container": dep.container_name,
                    "User": dep.username,
                    "VM": dep.vm_name,
                    "Image": dep.image_name,
                    "Status": dep.status.value,
                    "Created": str(dep.created_at)[:19] if dep.created_at else "",
                    "Last Check": str(dep.last_health_check)[:19] if dep.last_health_check else "Never",
                })

            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        except ImportError:
            # Fallback without pandas
            for dep in deployments:
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                    with col1:
                        st.markdown(f"**{dep.container_name}**")
                        st.caption(f"User: {dep.username}")

                    with col2:
                        st.text(f"VM: {dep.vm_name}")
                        st.text(f"Image: {dep.image_name}")

                    with col3:
                        status_icon = {"running": "🟢", "stopped": "🔴", "error": "🟠", "unknown": "⚪"}.get(dep.status.value, "⚪")
                        st.text(f"Status: {status_icon} {dep.status.value}")

                    with col4:
                        if dep.status == DeploymentStatus.RUNNING:
                            if st.button("Stop", key=f"admin_stop_{dep.id}"):
                                result = stop_container(dep.container_name, user_identity)
                                if result["success"]:
                                    db.update_deployment_status(dep.id, DeploymentStatus.STOPPED)
                                    st.rerun()

    st.divider()

    # User Management Section
    st.subheader("User Management")

    users = db.get_all_users()
    if users:
        try:
            import pandas as pd

            user_data = []
            for u in users:
                user_deployments = db.get_user_deployments(u.id, active_only=True)
                running_count = len([d for d in user_deployments if d.status == DeploymentStatus.RUNNING])
                user_data.append({
                    "Username": u.username,
                    "Email": u.email or "",
                    "Admin": "Yes" if u.is_admin else "No",
                    "Active Containers": running_count,
                    "Last Login": str(u.last_login)[:19] if u.last_login else "Never",
                })

            df = pd.DataFrame(user_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        except ImportError:
            for u in users:
                st.text(f"- {u.username} {'(Admin)' if u.is_admin else ''}")
    else:
        st.info("No users registered yet.")


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="Container Deployment Portal",
        page_icon="🐳",
        layout="wide",
    )

    st.title("🐳 Container Deployment Portal")

    # Initialize database
    db = get_db()

    # Authenticate
    user_identity = require_auth()

    # Get or create user in database
    db_user = User(
        username=user_identity.common_name,
        email=user_identity.email,
        organization=user_identity.organization,
        organizational_unit=user_identity.organizational_unit,
        is_admin=user_identity.is_admin,
    )
    db_user = db.get_or_create_user(db_user)

    # Show user info
    st.sidebar.success(f"Logged in: {user_identity.common_name}")
    st.sidebar.text(f"Email: {user_identity.email}")
    if user_identity.is_admin:
        st.sidebar.warning("Admin Access")

    # Main tabs - add Admin tab for admins
    if user_identity.is_admin:
        tab1, tab2, tab3, tab4 = st.tabs(["Deploy", "My Containers", "Admin Dashboard", "Help"])
    else:
        tab1, tab2, tab3 = st.tabs(["Deploy", "My Containers", "Help"])
        tab4 = None

    # === DEPLOY TAB ===
    with tab1:
        st.header("Deploy a New Container")

        # Load VM configuration
        try:
            vms = load_vm_config()
        except Exception as e:
            st.error(f"Failed to load VM configuration: {e}")
            vms = []

        # Project Configuration Section
        st.subheader("Project Configuration")

        with st.container(border=True):
            proj_col1, proj_col2 = st.columns(2)

            with proj_col1:
                project_name_raw = st.text_input(
                    "Project Name",
                    placeholder="My ML Project",
                    help="Spaces will be replaced with underscores"
                )

                if project_name_raw:
                    sanitized_name = sanitize_project_name(project_name_raw)
                    if sanitized_name != project_name_raw.lower().replace(' ', '_'):
                        st.caption(f"Will be saved as: `{sanitized_name}`")
                else:
                    sanitized_name = ""

            with proj_col2:
                vm_names = [vm.name for vm in vms]
                selected_vm_name = st.selectbox(
                    "Target VM",
                    options=vm_names if vm_names else ["No VMs configured"],
                    help="Select where to deploy the container"
                )
                selected_vm = next((vm for vm in vms if vm.name == selected_vm_name), None)

            default_root = selected_vm.root_directory if selected_vm else "/tmp"
            root_directory = st.text_input(
                "Root Directory",
                value=default_root,
                help="Base directory for project folders"
            )

            if sanitized_name and root_directory:
                project_path = f"{root_directory}/{sanitized_name}"
                st.info(f"**Project Path:** `{project_path}/`")
                st.caption("Directories created: `workspace/` (read-write), `data/` (read-only)")

        st.divider()

        col1, col2 = st.columns(2)

        with col2:
            st.subheader("Advanced Options")

            memory = st.select_slider(
                "Memory Limit",
                options=["8g", "16g", "32g", "64g", "128g"],
                value="32g",
            )

            gpu_enabled = st.checkbox("Enable GPU", value=True)

            st.info("""
            **Tips:**
            - GPU containers require NVIDIA drivers on the VM
            - Higher memory limits may require admin approval
            - Your workspace is persisted across container restarts
            """)

            st.divider()
            st.subheader("VM Status")

            if selected_vm:
                if st.button("Test Connection", key="test_vm_conn"):
                    with st.spinner(f"Testing connection to {selected_vm.name}..."):
                        ok, msg = test_vm_connection(selected_vm)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

                with st.expander("VM Details"):
                    st.text(f"Host: {selected_vm.host}")
                    if selected_vm.ssh_port:
                        st.text(f"SSH Port: {selected_vm.ssh_port}")
                        st.text(f"SSH User: {selected_vm.ssh_user}")
                    else:
                        st.text("Type: Local deployment")
                    st.text(f"Root: {selected_vm.root_directory}")
                    if selected_vm.description:
                        st.caption(selected_vm.description)

        with col1:
            st.subheader("Select Image")

            for img in CONFIG["curated_images"]:
                with st.container(border=True):
                    st.markdown(f"**{img['name']}**")
                    st.text(img["description"])
                    gpu_badge = "🟢 GPU" if img["gpu"] else "⚪ CPU"
                    st.text(f"{gpu_badge} | Ports: {', '.join(img['ports'].values())}")

                    deploy_disabled = not sanitized_name or not selected_vm

                    if st.button(
                        f"Deploy {img['name']}",
                        key=f"deploy_{img['name']}",
                        disabled=deploy_disabled
                    ):
                        with st.status("Deploying container...", expanded=True) as status:
                            def update_status(msg):
                                st.write(msg)

                            # Step 1: Test VM connection
                            update_status(f"Connecting to {selected_vm.name}...")
                            conn_ok, conn_msg = test_vm_connection(selected_vm)

                            if not conn_ok:
                                status.update(label="Deployment failed", state="error")
                                st.error(f"Connection failed: {conn_msg}")
                                continue

                            update_status(f"✅ {conn_msg}")

                            # Step 2: Create project directory
                            update_status(f"Creating project directory...")
                            dir_ok, dir_msg, project_path = create_project_directory(
                                selected_vm,
                                sanitized_name,
                                status_callback=update_status
                            )

                            if not dir_ok:
                                status.update(label="Deployment failed", state="error")
                                st.error(f"Directory creation failed: {dir_msg}")
                                continue

                            update_status(f"✅ {dir_msg}")

                            # Step 3: Deploy container
                            container_name = f"{sanitized_name}-{img['name']}"

                            user_index = abs(hash(user_identity.common_name)) % 500
                            port_start = CONFIG["user_port_start"] + (user_index * 100)

                            port_mappings = {}
                            port_offset = 0
                            for container_port, desc in img.get("ports", {}).items():
                                host_port = port_start + port_offset
                                port_mappings[str(host_port)] = container_port
                                port_offset += 1

                            full_image = f"{CONFIG['registry']}/{img['name']}:{img['tag']}"

                            update_status(f"Deploying {container_name}...")
                            result = deploy_container_remote(
                                vm=selected_vm,
                                project_path=project_path,
                                image=full_image,
                                container_name=container_name,
                                ports=port_mappings,
                                gpu=img["gpu"],
                                memory_limit=memory,
                                user_id=user_identity.common_name,
                                user_email=user_identity.email,
                                status_callback=update_status
                            )

                            if result["success"]:
                                status.update(label="Deployment complete!", state="complete")
                                update_status(f"✅ Container {container_name} is running")

                                # Save to database
                                deployment = Deployment(
                                    user_id=db_user.id,
                                    container_name=container_name,
                                    container_id=result.get("container_id"),
                                    vm_name=selected_vm.name,
                                    vm_host=selected_vm.host,
                                    project_path=project_path,
                                    image_name=img["name"],
                                    image_tag=img.get("tag", "latest"),
                                    ports=port_mappings,
                                    memory_limit=memory,
                                    gpu_enabled=img["gpu"],
                                    status=DeploymentStatus.RUNNING,
                                )
                                db.create_deployment(deployment)

                                # Legacy JSON log
                                log_deployment(user_identity, img["name"], container_name,
                                             {desc: port for port, desc in zip(port_mappings.keys(), img["ports"].values())})

                                st.success("Container deployed successfully!")

                                with st.container(border=True):
                                    st.markdown("**Access URLs:**")
                                    for port_desc, container_port in img["ports"].items():
                                        for host_port, cp in port_mappings.items():
                                            if cp == container_port:
                                                url = f"http://{selected_vm.host}:{host_port}"
                                                st.markdown(f"- **{port_desc}:** [{url}]({url})")
                                                break
                            else:
                                status.update(label="Deployment failed", state="error")
                                st.error(f"Deployment failed: {result.get('error', 'Unknown error')}")

                    if deploy_disabled and not sanitized_name:
                        st.caption("Enter a project name to enable deployment")

    # === MY CONTAINERS TAB ===
    with tab2:
        st.header("My Containers")

        # Get from database
        user_deployments = db.get_user_deployments(db_user.id, active_only=False)

        if not user_deployments:
            st.info("No containers deployed yet. Deploy one from the Deploy tab!")
        else:
            for dep in user_deployments:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([3, 2, 1])

                    with col1:
                        st.markdown(f"**{dep.container_name}**")
                        st.caption(f"Image: {dep.image_name} | VM: {dep.vm_name}")

                    with col2:
                        status_icon = {"running": "🟢", "stopped": "🔴", "error": "🟠", "unknown": "⚪", "removed": "⚫"}.get(dep.status.value, "⚪")
                        st.text(f"Status: {status_icon} {dep.status.value}")
                        if dep.ports:
                            ports_str = ", ".join(dep.ports.keys())
                            st.caption(f"Ports: {ports_str}")

                    with col3:
                        if dep.status == DeploymentStatus.RUNNING:
                            if st.button("Stop", key=f"stop_{dep.id}"):
                                result = stop_container(dep.container_name, user_identity)
                                if result["success"]:
                                    db.update_deployment_status(dep.id, DeploymentStatus.STOPPED)
                                    st.success("Stopped")
                                    st.rerun()
                                else:
                                    st.error(result["error"])

        if st.button("🔄 Refresh", key="my_containers_refresh"):
            st.rerun()

    # === ADMIN DASHBOARD TAB ===
    if user_identity.is_admin and tab4:
        with tab4:
            render_admin_dashboard(user_identity, db_user)

    # === HELP TAB ===
    help_tab = tab4 if not user_identity.is_admin else tab3
    if not user_identity.is_admin:
        help_tab = tab3

    with (tab4 if user_identity.is_admin else tab3):
        st.header("Help & Documentation")

        st.markdown("""
        ### Quick Start

        1. Enter a project name
        2. Select an image from the **Deploy** tab
        3. Click the **Deploy** button
        4. Copy the access URL and open in your browser
        5. Your workspace is automatically mounted at `/workspace`

        ### Available Images

        | Image | Description | GPU |
        |-------|-------------|-----|
        | gpu-jupyter | Full ML stack with Jupyter Lab | Yes |
        | gpu-pytorch | PyTorch-focused environment | Yes |
        | ml-cpu | Lightweight CPU-only environment | No |

        ### FAQ

        **Q: How do I save my work?**
        A: Everything in `/workspace` is persisted to your project directory.

        **Q: Can I install additional packages?**
        A: Yes, use `pip install --user <package>` or create a virtual environment.

        **Q: My container was stopped unexpectedly?**
        A: Check if you exceeded memory limits. Contact admin for higher limits.

        ### Support

        Contact your system administrator for help.
        """)


if __name__ == "__main__":
    main()
