# Administration Guide

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Image Management](#image-management)
3. [User VM Preparation](#user-vm-preparation)
4. [Deployment Service](#deployment-service)
5. [Monitoring & Maintenance](#monitoring--maintenance)
6. [Security Operations](#security-operations)
7. [Backup & Recovery](#backup--recovery)

---

## Initial Setup

### Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| Docker | 24.0+ | Container runtime |
| Docker Compose | 2.20+ | Multi-container orchestration |
| NVIDIA Driver | 535+ | GPU support on host |
| NVIDIA Container Toolkit | 1.14+ | GPU passthrough to containers |

### Install Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Verify
docker --version
docker run hello-world
```

### Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

## Image Management

### Building Curated Images

```bash
cd container-management/images/gpu-jupyter

# Build with default settings
docker build -t ghcr.io/your-org/gpu-jupyter:latest .

# Build with version tag
docker build -t ghcr.io/your-org/gpu-jupyter:1.0.0 .

# Build with custom user ID (for permission compatibility)
docker build \
    --build-arg USER_ID=1000 \
    --build-arg GROUP_ID=1000 \
    -t ghcr.io/your-org/gpu-jupyter:latest .
```

### Pushing to Registry

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Push image
docker push ghcr.io/your-org/gpu-jupyter:latest
docker push ghcr.io/your-org/gpu-jupyter:1.0.0

# Push all tags
docker push ghcr.io/your-org/gpu-jupyter --all-tags
```

### Image Versioning Strategy

```
ghcr.io/your-org/gpu-jupyter:latest      # Always current stable
ghcr.io/your-org/gpu-jupyter:1.0.0       # Specific version
ghcr.io/your-org/gpu-jupyter:1.0         # Minor version (auto-updates patch)
ghcr.io/your-org/gpu-jupyter:cuda12.2    # CUDA version variant
```

### Security Scanning

```bash
# Scan with Trivy (recommended)
trivy image ghcr.io/your-org/gpu-jupyter:latest

# Scan with Docker Scout
docker scout cves ghcr.io/your-org/gpu-jupyter:latest

# Generate SBOM (Software Bill of Materials)
docker sbom ghcr.io/your-org/gpu-jupyter:latest > sbom.json
```

### Updating Images

```bash
# 1. Update requirements.txt with new versions
# 2. Test locally
docker build -t gpu-jupyter:test .
docker run --gpus all -it gpu-jupyter:test python -c "import torch; print(torch.cuda.is_available())"

# 3. Tag and push
docker tag gpu-jupyter:test ghcr.io/your-org/gpu-jupyter:1.0.1
docker push ghcr.io/your-org/gpu-jupyter:1.0.1

# 4. Update :latest tag
docker tag ghcr.io/your-org/gpu-jupyter:1.0.1 ghcr.io/your-org/gpu-jupyter:latest
docker push ghcr.io/your-org/gpu-jupyter:latest
```

---

## User VM Preparation

### VM Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32+ GB |
| Disk | 100 GB | 500+ GB SSD |
| GPU | 1x NVIDIA | Per user needs |
| Network | 1 Gbps | 10 Gbps |

### Automated VM Setup

```bash
#!/bin/bash
# scripts/setup-vm.sh

VM_HOST=$1

ssh $VM_HOST << 'EOF'
    # Install Docker
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker $USER
    
    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    # Verify
    docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
    
    # Create standard directories
    mkdir -p ~/workspace ~/datasets
    
    # Pull curated images
    docker pull ghcr.io/your-org/gpu-jupyter:latest
EOF
```

### Port Allocation

Assign each user a unique port range:

```bash
# /etc/container-ports.conf
# Format: USERNAME:START_PORT:END_PORT

jsmith:10000:10099
ajones:10100:10199
bwilson:10200:10299
```

### User Account Setup

```bash
# Create user with Docker access
sudo useradd -m -s /bin/bash -G docker $USERNAME

# Create workspace directories
sudo mkdir -p /home/$USERNAME/workspace
sudo mkdir -p /home/$USERNAME/.jupyter
sudo chown -R $USERNAME:$USERNAME /home/$USERNAME
```

---

## Deployment Service

### Streamlit Service Setup

```bash
cd container-management/streamlit-service

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your settings

# Run (development)
streamlit run app.py

# Run (production with Docker)
docker-compose -f docker-compose.streamlit.yml up -d
```

### PKI Configuration

```python
# streamlit-service/auth.py configuration

PKI_CONFIG = {
    "ca_cert_path": "/etc/pki/ca-bundle.crt",
    "verify_crl": True,
    "crl_path": "/etc/pki/crl.pem",
    "required_ous": ["DataScience", "Research"],  # Allowed organizational units
    "admin_cns": ["admin1", "admin2"],  # Admin user common names
}
```

### Service Health Monitoring

```bash
# Check Streamlit service
curl -f http://localhost:8501/healthz

# Check all user containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check GPU utilization across containers
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

---

## Monitoring & Maintenance

### Container Status

```bash
# List all running containers
docker ps

# List containers for specific user
docker ps --filter "name=jupyter-jsmith"

# Resource usage
docker stats

# Detailed container info
docker inspect jupyter-jsmith
```

### Log Management

```bash
# View container logs
docker logs jupyter-jsmith

# Follow logs in real-time
docker logs -f jupyter-jsmith

# Logs with timestamps
docker logs -t jupyter-jsmith

# Last 100 lines
docker logs --tail 100 jupyter-jsmith
```

### Cleanup Operations

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes (CAUTION: may delete user data)
docker volume prune

# Full cleanup
docker system prune -a --volumes

# Scheduled cleanup (cron)
# 0 2 * * 0 docker system prune -f >> /var/log/docker-cleanup.log 2>&1
```

### GPU Monitoring

```bash
# Real-time GPU status
watch -n 1 nvidia-smi

# GPU utilization history
nvidia-smi dmon -s u

# Per-process GPU memory
nvidia-smi pmon -s m

# Log GPU metrics
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 60 >> /var/log/gpu-metrics.csv
```

---

## Security Operations

### Image Verification

```bash
# Verify image digest
docker pull ghcr.io/your-org/gpu-jupyter:latest
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/your-org/gpu-jupyter:latest

# Compare with known-good digest
EXPECTED_DIGEST="sha256:abc123..."
ACTUAL_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/your-org/gpu-jupyter:latest | cut -d@ -f2)
[ "$EXPECTED_DIGEST" = "$ACTUAL_DIGEST" ] && echo "VERIFIED" || echo "MISMATCH"
```

### Access Audit

```bash
# Container access log
docker events --filter 'type=container' --format '{{.Time}} {{.Action}} {{.Actor.Attributes.name}}'

# Parse Jupyter access logs
docker logs jupyter-jsmith 2>&1 | grep -E "GET|POST" 
```

### Network Isolation Verification

```bash
# Check container network settings
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' jupyter-jsmith

# Verify port bindings
docker port jupyter-jsmith

# Test isolation (from container)
docker exec jupyter-jsmith ping -c 1 internal-server.local  # Should fail if isolated
```

---

## Backup & Recovery

### User Data Backup

```bash
# Backup user workspace
tar -czvf backup-jsmith-$(date +%Y%m%d).tar.gz /home/jsmith/workspace

# Backup Jupyter config
docker run --rm -v jupyter_config:/data -v $(pwd):/backup alpine tar -czvf /backup/jupyter-config.tar.gz /data
```

### Container State Export

```bash
# Export container as image (includes changes)
docker commit jupyter-jsmith jupyter-jsmith-backup:$(date +%Y%m%d)

# Save image to file
docker save jupyter-jsmith-backup:$(date +%Y%m%d) > jupyter-jsmith-backup.tar
```

### Disaster Recovery

```bash
# Restore from backup
tar -xzvf backup-jsmith-20240115.tar.gz -C /

# Restore container from saved image
docker load < jupyter-jsmith-backup.tar
docker run --gpus all -d -p 10000:8888 jupyter-jsmith-backup:20240115
```

---

## Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| Start container | `docker-compose up -d` |
| Stop container | `docker-compose down` |
| View logs | `docker logs -f <container>` |
| Shell access | `docker exec -it <container> /bin/bash` |
| Resource usage | `docker stats` |
| GPU status | `nvidia-smi` |
| Restart container | `docker restart <container>` |
| Pull latest image | `docker pull ghcr.io/your-org/gpu-jupyter:latest` |

### Emergency Procedures

| Situation | Action |
|-----------|--------|
| Container consuming too much memory | `docker update --memory 8g <container>` |
| GPU frozen | `sudo nvidia-smi --gpu-reset` |
| Container unresponsive | `docker kill <container>` |
| Disk full | `docker system prune -a` |
| Network issues | `docker network disconnect/connect` |
