#!/bin/bash
# =============================================================================
# VM SETUP SCRIPT
# =============================================================================
#
# Prepares a VM to run GPU containers with Docker.
#
# Usage:
#   ./setup-vm.sh              # Run on local VM
#   ssh user@vm ./setup-vm.sh  # Run on remote VM
#
# Prerequisites:
#   - Ubuntu 22.04 or similar
#   - NVIDIA GPU with driver already installed (check: nvidia-smi)
#   - sudo access
#
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo " GPU Container VM Setup"
echo "=========================================="

# -----------------------------------------------------------------------------
# CHECK PREREQUISITES
# -----------------------------------------------------------------------------

echo ""
echo "[1/6] Checking prerequisites..."

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
    echo "  Re-running with sudo..."
    exec sudo "$0" "$@"
fi

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA driver not found. Install driver first:"
    echo "   sudo apt install nvidia-driver-535"
    exit 1
fi

echo "✅ NVIDIA driver found:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1

# -----------------------------------------------------------------------------
# INSTALL DOCKER
# -----------------------------------------------------------------------------

echo ""
echo "[2/6] Installing Docker..."

if command -v docker &> /dev/null; then
    echo "✅ Docker already installed: $(docker --version)"
else
    curl -fsSL https://get.docker.com | sh
    echo "✅ Docker installed"
fi

# Add current user to docker group
SUDO_USER=${SUDO_USER:-$USER}
usermod -aG docker $SUDO_USER
echo "  Added $SUDO_USER to docker group"

# -----------------------------------------------------------------------------
# INSTALL DOCKER COMPOSE
# -----------------------------------------------------------------------------

echo ""
echo "[3/6] Installing Docker Compose..."

if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose already installed: $(docker-compose --version)"
else
    apt-get update
    apt-get install -y docker-compose-plugin
    echo "✅ Docker Compose installed"
fi

# -----------------------------------------------------------------------------
# INSTALL NVIDIA CONTAINER TOOLKIT
# -----------------------------------------------------------------------------

echo ""
echo "[4/6] Installing NVIDIA Container Toolkit..."

if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "✅ NVIDIA Container Toolkit already installed"
else
    # Add NVIDIA repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        tee /etc/apt/sources.list.d/nvidia-docker.list
    
    apt-get update
    apt-get install -y nvidia-container-toolkit
    
    echo "✅ NVIDIA Container Toolkit installed"
fi

# Configure Docker to use NVIDIA runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "✅ Docker configured for NVIDIA runtime"

# -----------------------------------------------------------------------------
# VERIFY GPU ACCESS
# -----------------------------------------------------------------------------

echo ""
echo "[5/6] Verifying GPU access in containers..."

if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU accessible from containers"
else
    echo "❌ GPU not accessible from containers"
    echo "   Check: nvidia-ctk runtime configure --runtime=docker"
    exit 1
fi

# -----------------------------------------------------------------------------
# CREATE STANDARD DIRECTORIES
# -----------------------------------------------------------------------------

echo ""
echo "[6/6] Creating standard directories..."

# Shared datasets directory
mkdir -p /data/shared/datasets
chmod 755 /data/shared

# User workspace template
mkdir -p /etc/skel/workspace
mkdir -p /etc/skel/.jupyter

echo "✅ Directories created"

# -----------------------------------------------------------------------------
# PULL BASE IMAGES (OPTIONAL)
# -----------------------------------------------------------------------------

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""
echo "Docker Info:"
docker --version
docker compose version
echo ""
echo "Next steps:"
echo "  1. Pull curated images:  docker pull ghcr.io/your-org/gpu-jupyter:latest"
echo "  2. Create user accounts"
echo "  3. Test container deployment"
echo ""
echo "Test GPU container:"
echo "  docker run --gpus all -p 8888:8888 ghcr.io/your-org/gpu-jupyter:latest"
echo ""
