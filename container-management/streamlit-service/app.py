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
import streamlit as st

from remote_deploy import (
    load_vm_config, deploy_container_remote,
    create_project_directory, sanitize_project_name,
    get_container_status_remote, stop_container_remote,
    list_containers_remote, test_vm_connection, VMConfig
)

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
    "user_port_start": 10000,  # Each user gets 100 ports starting from user_index * 100
    "vm_host": os.getenv("VM_HOST", "localhost"),
    "log_file": os.path.join(os.path.expanduser("~"), ".container-deployments.log"),
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
    """
    Extract user identity from PKI certificate headers.
    
    In production, the reverse proxy (nginx/Apache) validates the certificate
    and passes the DN components as headers:
        - SSL_CLIENT_S_DN_CN: Common Name
        - SSL_CLIENT_S_DN_Email: Email
        - SSL_CLIENT_S_DN_O: Organization
        - SSL_CLIENT_S_DN_OU: Organizational Unit
    """
    # In development, use environment variable override
    if os.getenv("DEV_USER"):
        return UserIdentity(
            common_name=os.getenv("DEV_USER"),
            email=f"{os.getenv('DEV_USER')}@dev.local",
            organization="Development",
            organizational_unit="DataScience",
            is_admin=os.getenv("DEV_ADMIN", "false").lower() == "true",
        )
    
    # Production: get from headers (set by reverse proxy)
    headers = st.context.headers if hasattr(st.context, 'headers') else {}
    
    cn = headers.get("SSL_CLIENT_S_DN_CN") or headers.get("X-SSL-Client-CN")
    if not cn:
        return None
    
    return UserIdentity(
        common_name=cn,
        email=headers.get("SSL_CLIENT_S_DN_Email", f"{cn}@unknown"),
        organization=headers.get("SSL_CLIENT_S_DN_O", "Unknown"),
        organizational_unit=headers.get("SSL_CLIENT_S_DN_OU", "Unknown"),
        is_admin=cn in CONFIG.get("admin_users", []),
    )


def require_auth() -> UserIdentity:
    """Require valid PKI authentication or show error."""
    user = get_user_from_pki()
    if not user:
        st.error("⚠️ Authentication required. Please ensure your CAC/PIV is inserted.")
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


def get_running_containers(user: str = None) -> List[Dict]:
    """Get list of running containers, optionally filtered by user."""
    result = run_docker_command([
        "ps", "--format", 
        '{"name":"{{.Names}}","image":"{{.Image}}","status":"{{.Status}}","ports":"{{.Ports}}"}'
    ])
    
    containers = []
    for line in result.stdout.strip().split("\n"):
        if line:
            try:
                container = json.loads(line)
                if user is None or container["name"].endswith(f"-{user}"):
                    containers.append(container)
            except json.JSONDecodeError:
                continue
    return containers


def deploy_container(
    image_name: str,
    user: UserIdentity,
    gpu: bool = True,
    memory_limit: str = "32g",
) -> Dict:
    """
    Deploy a container for a user.
    
    Returns dict with deployment status and access URLs.
    """
    # Generate container name
    container_name = f"{image_name}-{user.common_name}"
    
    # Get image config
    image_config = next(
        (img for img in CONFIG["curated_images"] if img["name"] == image_name),
        None
    )
    if not image_config:
        return {"success": False, "error": f"Unknown image: {image_name}"}
    
    # Get user port range (simplified - in production, use database)
    # Limit to 500 users to keep ports under 65535 (10000 + 500*100 = 60000)
    user_index = abs(hash(user.common_name)) % 500
    port_start, _ = get_user_port_range(user_index)
    
    # Build docker run command
    cmd = [
        "run", "-d",
        "--name", container_name,
        "--memory", memory_limit,
        "--shm-size", "2g",
        "-e", f"USER_ID={user.common_name}",
        "-e", f"USER_EMAIL={user.email}",
    ]
    
    # Add GPU flag if requested
    if gpu and image_config.get("gpu"):
        cmd.extend(["--gpus", "all"])
    
    # Add port mappings
    port_offset = 0
    port_mappings = {}
    for container_port, description in image_config.get("ports", {}).items():
        host_port = port_start + port_offset
        cmd.extend(["-p", f"{host_port}:{container_port}"])
        port_mappings[description] = host_port
        port_offset += 1
    
    # Add volume mounts
    workspace_path = f"/home/{user.common_name}/workspace"
    cmd.extend(["-v", f"{workspace_path}:/workspace"])
    
    # Add image
    full_image = f"{CONFIG['registry']}/{image_name}:{image_config['tag']}"
    cmd.append(full_image)
    
    # Stop existing container if running
    run_docker_command(["stop", container_name])
    run_docker_command(["rm", container_name])
    
    # Deploy
    result = run_docker_command(cmd)
    
    if result.returncode != 0:
        return {
            "success": False,
            "error": result.stderr,
        }
    
    # Log deployment
    log_deployment(user, image_name, container_name, port_mappings)
    
    return {
        "success": True,
        "container_name": container_name,
        "ports": port_mappings,
        "host": CONFIG["vm_host"],
    }


def stop_container(container_name: str, user: UserIdentity) -> Dict:
    """Stop a user's container."""
    # Verify container belongs to user
    if not container_name.endswith(f"-{user.common_name}") and not user.is_admin:
        return {"success": False, "error": "Permission denied"}
    
    result = run_docker_command(["stop", container_name])
    if result.returncode != 0:
        return {"success": False, "error": result.stderr}
    
    return {"success": True}


def log_deployment(user: UserIdentity, image: str, container: str, ports: Dict):
    """Log deployment for audit trail."""
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
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="Container Deployment Portal",
        page_icon="🐳",
        layout="wide",
    )
    
    st.title("🐳 GPU Container Deployment Portal")
    
    # Authenticate
    user = require_auth()
    
    # Show user info
    st.sidebar.success(f"✅ Authenticated: {user.common_name}")
    st.sidebar.text(f"Email: {user.email}")
    st.sidebar.text(f"OU: {user.organizational_unit}")
    if user.is_admin:
        st.sidebar.warning("🔑 Admin Access")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Deploy", "My Containers", "Help"])
    
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

                # Show sanitized version
                if project_name_raw:
                    sanitized_name = sanitize_project_name(project_name_raw)
                    if sanitized_name != project_name_raw.lower().replace(' ', '_'):
                        st.caption(f"Will be saved as: `{sanitized_name}`")
                else:
                    sanitized_name = ""

            with proj_col2:
                # VM selector
                vm_names = [vm.name for vm in vms]
                selected_vm_name = st.selectbox(
                    "Target VM",
                    options=vm_names if vm_names else ["No VMs configured"],
                    help="Select where to deploy the container"
                )

                # Get selected VM config
                selected_vm = next((vm for vm in vms if vm.name == selected_vm_name), None)

            # Root directory (auto-filled from VM, editable)
            default_root = selected_vm.root_directory if selected_vm else "/tmp"
            root_directory = st.text_input(
                "Root Directory",
                value=default_root,
                help="Base directory for project folders"
            )

            # Show project path preview
            if sanitized_name and root_directory:
                project_path = f"{root_directory}/{sanitized_name}"
                st.info(f"**Project Path:** `{project_path}/`")
                st.caption("Directories created: `workspace/` (read-write), `data/` (read-only)")

        st.divider()

        col1, col2 = st.columns(2)

        # Define advanced options FIRST so they're available for deployment
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

            # VM Connection Test
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

                # Show VM details
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

                    # Disable button if no project name
                    deploy_disabled = not sanitized_name or not selected_vm

                    if st.button(
                        f"Deploy {img['name']}",
                        key=f"deploy_{img['name']}",
                        disabled=deploy_disabled
                    ):
                        # Create status expander for real-time feedback
                        with st.status("Deploying container...", expanded=True) as status:
                            status_messages = []

                            def update_status(msg):
                                status_messages.append(msg)
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

                            # Calculate ports
                            user_index = abs(hash(user.common_name)) % 500
                            port_start = CONFIG["user_port_start"] + (user_index * 100)

                            port_mappings = {}
                            port_offset = 0
                            for container_port, desc in img.get("ports", {}).items():
                                host_port = port_start + port_offset
                                port_mappings[str(host_port)] = container_port
                                port_offset += 1

                            # Full image path
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
                                user_id=user.common_name,
                                user_email=user.email,
                                status_callback=update_status
                            )

                            if result["success"]:
                                status.update(label="Deployment complete!", state="complete")
                                update_status(f"✅ Container {container_name} is running")

                                # Log deployment
                                log_deployment(user, img["name"], container_name,
                                             {desc: port for port, desc in zip(port_mappings.keys(), img["ports"].values())})

                                # Show access URLs
                                st.success("Container deployed successfully!")

                                with st.container(border=True):
                                    st.markdown("**Access URLs:**")
                                    for port_desc, container_port in img["ports"].items():
                                        # Find the host port for this container port
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
        st.header("My Running Containers")
        
        containers = get_running_containers(user.common_name)
        
        if not containers:
            st.info("No running containers. Deploy one from the Deploy tab!")
        else:
            for container in containers:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{container['name']}**")
                        st.text(f"Image: {container['image']}")
                    
                    with col2:
                        st.text(f"Status: {container['status']}")
                        st.text(f"Ports: {container['ports']}")
                    
                    with col3:
                        if st.button("Stop", key=f"stop_{container['name']}"):
                            result = stop_container(container["name"], user)
                            if result["success"]:
                                st.success("Stopped")
                                st.rerun()
                            else:
                                st.error(result["error"])
        
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # === HELP TAB ===
    with tab3:
        st.header("Help & Documentation")
        
        st.markdown("""
        ### Quick Start
        
        1. Select an image from the **Deploy** tab
        2. Click the **Deploy** button
        3. Copy the access URL and open in your browser
        4. Your workspace is automatically mounted at `/workspace`
        
        ### Available Images
        
        | Image | Description | GPU |
        |-------|-------------|-----|
        | gpu-jupyter | Full ML stack with Jupyter Lab | ✅ |
        | gpu-pytorch | PyTorch-focused environment | ✅ |
        | ml-cpu | Lightweight CPU-only environment | ❌ |
        
        ### FAQ
        
        **Q: How do I save my work?**
        A: Everything in `/workspace` is persisted to your home directory.
        
        **Q: Can I install additional packages?**
        A: Yes, use `pip install --user <package>` or create a virtual environment.
        
        **Q: My container was stopped unexpectedly?**
        A: Check if you exceeded memory limits. Contact admin for higher limits.
        
        ### Support
        
        Contact: container-support@your-org.com
        """)


if __name__ == "__main__":
    main()
