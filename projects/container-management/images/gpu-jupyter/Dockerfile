# =============================================================================
# GPU-ENABLED JUPYTER NOTEBOOK CONTAINER
# =============================================================================
# 
# PURPOSE:
#   Provides a Jupyter Lab environment with GPU access for data science work.
#   Includes PyTorch, TensorFlow, and common ML libraries.
#
# BASE IMAGE:
#   NVIDIA CUDA runtime - official NVIDIA image with GPU drivers and libraries.
#   Using 'runtime' (not 'devel') to minimize size while retaining GPU capability.
#
# SECURITY:
#   - Base image from official NVIDIA NGC registry
#   - All package versions pinned for reproducibility
#   - No secrets or credentials embedded
#   - Runs as non-root user (configurable)
#
# CUSTOMIZATION POINTS:
#   - Line 30: CUDA version (match your GPU driver compatibility)
#   - Line 45: Python version
#   - Line 78: Python packages (requirements.txt)
#   - Line 95: Jupyter configuration
#   - Line 105: Exposed ports
#
# =============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: BASE IMAGE SELECTION
# -----------------------------------------------------------------------------
# 
# NVIDIA provides several image variants:
#   - base:    Minimal, just CUDA driver (~120MB)
#   - runtime: CUDA libraries included (~3.5GB) ← WE USE THIS
#   - devel:   Includes compilers, headers (~6GB)
#
# Match CUDA version to your GPU driver:
#   - Driver 525.x → CUDA 12.0
#   - Driver 530.x → CUDA 12.1
#   - Driver 535.x → CUDA 12.2
#
# Check driver version: nvidia-smi
#
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# -----------------------------------------------------------------------------
# METADATA
# -----------------------------------------------------------------------------
# Labels help identify images and track provenance.
#
LABEL maintainer="your-team@your-org.com"
LABEL version="1.0.0"
LABEL description="GPU-enabled Jupyter Lab for data science"
LABEL org.opencontainers.image.source="https://github.com/your-org/container-management"

# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
#
# DEBIAN_FRONTEND: Prevents interactive prompts during apt-get
# PYTHONUNBUFFERED: Ensures print() statements appear in logs immediately
# PYTHONDONTWRITEBYTECODE: Prevents .pyc files (cleaner container)
# CUDA_HOME: Required by some CUDA-dependent packages
# PATH: Ensures local pip installs are accessible
#
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/home/jupyter/.local/bin:${PATH}"

# -----------------------------------------------------------------------------
# SYSTEM PACKAGES
# -----------------------------------------------------------------------------
#
# Install OS-level dependencies:
#   - python3.11: Specific Python version (not system default)
#   - python3-pip: Package installer
#   - git: For pip installing from git repos
#   - curl, wget: For downloading files
#   - build-essential: Compiler toolchain (needed for some pip packages)
#   - libgl1: OpenCV dependency
#   - libglib2.0-0: Common library dependency
#
# SECURITY NOTE: 
#   rm -rf /var/lib/apt/lists/* removes package cache, reducing image size
#   and eliminating potential information disclosure.
#
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# -----------------------------------------------------------------------------
# PYTHON CONFIGURATION
# -----------------------------------------------------------------------------
#
# Set Python 3.11 as the default 'python' command.
# update-alternatives manages multiple versions of the same command.
#
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip to latest version for security and compatibility
RUN python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# NON-ROOT USER (Optional but recommended)
# -----------------------------------------------------------------------------
#
# Running as non-root improves security:
#   - Limits container's ability to modify host if escape occurs
#   - Matches typical HPC/shared cluster configurations
#
# To run as root instead, comment out this section and the USER directive.
#
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=jupyter

RUN groupadd -g ${GROUP_ID} ${USERNAME} \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}

# -----------------------------------------------------------------------------
# PYTHON PACKAGES
# -----------------------------------------------------------------------------
#
# COPY requirements.txt first (before other files) for Docker layer caching.
# If requirements.txt doesn't change, this layer is cached on rebuilds.
#
# --no-cache-dir: Don't store pip's cache (reduces image size)
#
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# -----------------------------------------------------------------------------
# JUPYTER CONFIGURATION
# -----------------------------------------------------------------------------
#
# Create Jupyter config directory and set up default configuration.
#
# Configuration options:
#   - ip='*': Listen on all interfaces (required for Docker)
#   - port=8888: Standard Jupyter port
#   - allow_root: Only if running as root (not recommended)
#   - token='': Disable token auth (use external auth instead)
#              OR set a token for standalone use
#
RUN mkdir -p /home/${USERNAME}/.jupyter

# Generate Jupyter config
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /home/${USERNAME}/.jupyter/jupyter_server_config.py \
    && echo "c.ServerApp.port = 8888" >> /home/${USERNAME}/.jupyter/jupyter_server_config.py \
    && echo "c.ServerApp.open_browser = False" >> /home/${USERNAME}/.jupyter/jupyter_server_config.py \
    && echo "c.ServerApp.allow_origin = '*'" >> /home/${USERNAME}/.jupyter/jupyter_server_config.py

# Fix ownership if using non-root user
RUN chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

# -----------------------------------------------------------------------------
# WORKING DIRECTORY
# -----------------------------------------------------------------------------
#
# /workspace is mounted from host, providing persistent storage.
# Data here survives container restarts.
#
WORKDIR /workspace

# -----------------------------------------------------------------------------
# SWITCH TO NON-ROOT USER
# -----------------------------------------------------------------------------
#
# All subsequent commands (and the running container) use this user.
# Comment out to run as root.
#
USER ${USERNAME}

# -----------------------------------------------------------------------------
# EXPOSED PORTS
# -----------------------------------------------------------------------------
#
# EXPOSE documents which ports the container listens on.
# Does NOT publish them - that's done at runtime with -p flag.
#
# Ports:
#   8888: Jupyter Lab
#   6006: TensorBoard
#
EXPOSE 8888 6006

# -----------------------------------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------------------------------
#
# Docker can automatically check if the container is healthy.
# This helps orchestration systems detect and restart failed containers.
#
# --interval: Check every 30 seconds
# --timeout: Fail if check takes >10 seconds
# --retries: Mark unhealthy after 3 failures
#
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# -----------------------------------------------------------------------------
# STARTUP COMMAND
# -----------------------------------------------------------------------------
#
# CMD specifies the default command when container starts.
# Can be overridden at runtime: docker run <image> <different-command>
#
# Options:
#   --ip=0.0.0.0: Bind to all interfaces (required for Docker networking)
#   --port=8888: Listen on port 8888
#   --no-browser: Don't try to open browser (no GUI in container)
#   --NotebookApp.token='': Disable token (use external auth)
#                           OR set token for standalone security
#
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

# =============================================================================
# BUILD INSTRUCTIONS
# =============================================================================
#
# Build:
#   docker build -t gpu-jupyter:latest .
#
# Build with custom user ID (match host user for volume permissions):
#   docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t gpu-jupyter:latest .
#
# =============================================================================
# RUN INSTRUCTIONS
# =============================================================================
#
# Basic run with GPU:
#   docker run --gpus all -p 8888:8888 gpu-jupyter:latest
#
# With persistent storage:
#   docker run --gpus all -p 8888:8888 -v $(pwd):/workspace gpu-jupyter:latest
#
# With specific GPU:
#   docker run --gpus '"device=0"' -p 8888:8888 gpu-jupyter:latest
#
# With TensorBoard port:
#   docker run --gpus all -p 8888:8888 -p 6006:6006 gpu-jupyter:latest
#
# Interactive shell instead of Jupyter:
#   docker run --gpus all -it gpu-jupyter:latest /bin/bash
#
# =============================================================================
