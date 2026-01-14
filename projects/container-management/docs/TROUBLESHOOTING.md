# Troubleshooting Guide

## Table of Contents

1. [GPU Issues](#gpu-issues)
2. [Container Startup Failures](#container-startup-failures)
3. [Network Problems](#network-problems)
4. [Permission Errors](#permission-errors)
5. [Resource Constraints](#resource-constraints)
6. [Jupyter-Specific Issues](#jupyter-specific-issues)
7. [Docker Compose Issues](#docker-compose-issues)
8. [Diagnostic Commands](#diagnostic-commands)

---

## GPU Issues

### GPU Not Visible in Container

**Symptom:** `nvidia-smi` works on host but fails in container, or PyTorch reports `cuda.is_available() = False`

**Diagnosis:**
```bash
# Check host GPU
nvidia-smi

# Check NVIDIA Container Toolkit
dpkg -l | grep nvidia-container

# Check Docker runtime configuration
docker info | grep -i runtime
```

**Solutions:**

1. **NVIDIA Container Toolkit not installed:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2. **Docker not configured for NVIDIA runtime:**
```bash
# Check /etc/docker/daemon.json contains:
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}

# Restart Docker
sudo systemctl restart docker
```

3. **Container not started with GPU flag:**
```bash
# Correct way
docker run --gpus all ...

# Or in docker-compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### CUDA Version Mismatch

**Symptom:** `CUDA error: no kernel image is available for execution on the device`

**Diagnosis:**
```bash
# Host CUDA driver version
nvidia-smi | grep "CUDA Version"

# Container CUDA version
docker run --gpus all <image> nvcc --version
```

**Solution:** Match container CUDA version to driver capability:

| Driver Version | Max CUDA Version |
|----------------|------------------|
| 525.x | 12.0 |
| 530.x | 12.1 |
| 535.x | 12.2 |
| 545.x | 12.3 |

Rebuild image with appropriate base:
```dockerfile
# For driver 535.x, use CUDA 12.2
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
```

### GPU Out of Memory

**Symptom:** `CUDA out of memory` error

**Diagnosis:**
```bash
# Check GPU memory usage
nvidia-smi

# Check which processes are using GPU
nvidia-smi pmon -s m
```

**Solutions:**

1. **Clear GPU memory:**
```python
import torch
torch.cuda.empty_cache()
```

2. **Reduce batch size in training**

3. **Kill other GPU processes:**
```bash
# Find PID from nvidia-smi, then:
kill -9 <PID>
```

4. **Limit container GPU memory:**
```bash
docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=0 \
    -e CUDA_MEMORY_FRACTION=0.5 ...
```

---

## Container Startup Failures

### Container Exits Immediately

**Symptom:** Container starts then exits with code 0 or 1

**Diagnosis:**
```bash
# Check exit code
docker ps -a --filter "name=jupyter" --format "{{.Status}}"

# Check logs
docker logs jupyter-username

# Check last 50 lines
docker logs --tail 50 jupyter-username
```

**Common Causes:**

1. **No foreground process:**
```dockerfile
# Wrong - script exits immediately
CMD ["./start.sh"]

# Right - keeps running
CMD ["jupyter", "lab", "--ip=0.0.0.0"]
```

2. **Missing dependencies:**
```bash
# Check for import errors
docker run --rm <image> python -c "import torch; import transformers"
```

3. **Permission issues on mounted volumes:**
```bash
# Check ownership
ls -la /path/to/mounted/volume

# Fix
sudo chown -R 1000:1000 /path/to/mounted/volume
```

### Port Already in Use

**Symptom:** `Bind for 0.0.0.0:8888 failed: port is already allocated`

**Diagnosis:**
```bash
# Find what's using the port
sudo lsof -i :8888
# or
sudo netstat -tlnp | grep 8888
```

**Solutions:**

1. **Stop conflicting container:**
```bash
docker stop <container-using-port>
```

2. **Use different port:**
```bash
docker run -p 8889:8888 ...
```

3. **Kill process using port:**
```bash
sudo kill -9 $(sudo lsof -t -i:8888)
```

---

## Network Problems

### Cannot Access Jupyter from Browser

**Symptom:** Connection refused or timeout when accessing `http://host:8888`

**Diagnosis:**
```bash
# Check container is running
docker ps | grep jupyter

# Check port mapping
docker port jupyter-username

# Check Jupyter is listening
docker exec jupyter-username netstat -tlnp | grep 8888

# Check firewall
sudo ufw status
sudo iptables -L -n | grep 8888
```

**Solutions:**

1. **Jupyter not binding to all interfaces:**
```dockerfile
# Must use 0.0.0.0, not localhost or 127.0.0.1
CMD ["jupyter", "lab", "--ip=0.0.0.0"]
```

2. **Firewall blocking port:**
```bash
sudo ufw allow 8888/tcp
# or
sudo iptables -A INPUT -p tcp --dport 8888 -j ACCEPT
```

3. **Port not published:**
```bash
# Must include -p flag
docker run -p 8888:8888 ...
```

### Container Cannot Reach External Network

**Symptom:** `pip install` or `git clone` fails with network error

**Diagnosis:**
```bash
# Test from inside container
docker exec jupyter-username ping -c 3 8.8.8.8
docker exec jupyter-username curl -I https://pypi.org
```

**Solutions:**

1. **DNS resolution failure:**
```bash
# Add DNS servers
docker run --dns 8.8.8.8 ...

# Or in docker-compose.yml
dns:
  - 8.8.8.8
  - 8.8.4.4
```

2. **Proxy required:**
```bash
docker run \
    -e HTTP_PROXY=http://proxy.corp:8080 \
    -e HTTPS_PROXY=http://proxy.corp:8080 \
    ...
```

3. **Network mode issue:**
```bash
# Try host networking (less isolated but may fix issues)
docker run --network host ...
```

---

## Permission Errors

### Cannot Write to Mounted Volume

**Symptom:** `Permission denied` when saving files in `/workspace`

**Diagnosis:**
```bash
# Check container user
docker exec jupyter-username id

# Check mount permissions
docker exec jupyter-username ls -la /workspace
```

**Solutions:**

1. **Match container user to host user:**
```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t gpu-jupyter .
```

2. **Fix host directory permissions:**
```bash
sudo chown -R $(id -u):$(id -g) /path/to/workspace
```

3. **Run container as root (not recommended):**
```bash
docker run --user root ...
```

### Cannot Install Packages

**Symptom:** `pip install` fails with permission error

**Diagnosis:**
```bash
# Check Python path
docker exec jupyter-username which python
docker exec jupyter-username python -m site --user-site
```

**Solutions:**

1. **Install to user directory:**
```bash
pip install --user <package>
```

2. **Use virtual environment:**
```bash
python -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install <package>
```

---

## Resource Constraints

### Container Killed (OOMKilled)

**Symptom:** Container exits suddenly, `docker inspect` shows `OOMKilled: true`

**Diagnosis:**
```bash
docker inspect jupyter-username | grep -i oom
docker stats --no-stream jupyter-username
```

**Solutions:**

1. **Increase memory limit:**
```bash
docker update --memory 64g --memory-swap 64g jupyter-username

# Or in docker-compose.yml
mem_limit: 64g
```

2. **Add swap:**
```bash
# Create swap file on host
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Shared Memory Error (PyTorch DataLoader)

**Symptom:** `RuntimeError: DataLoader worker is killed by signal: Bus error` or `unable to open shared memory`

**Diagnosis:**
```bash
# Check shared memory size
docker exec jupyter-username df -h /dev/shm
```

**Solution:**
```bash
# Increase shared memory
docker run --shm-size=2g ...

# Or in docker-compose.yml
shm_size: '2g'
```

---

## Jupyter-Specific Issues

### Jupyter Token Issues

**Symptom:** Cannot login, "Token authentication is enabled"

**Diagnosis:**
```bash
# Find token in logs
docker logs jupyter-username 2>&1 | grep token
```

**Solutions:**

1. **Get token from logs:**
```bash
docker logs jupyter-username 2>&1 | grep -o 'token=[a-zA-Z0-9]*' | head -1
```

2. **Disable token (use external auth):**
```bash
docker run ... jupyter lab --NotebookApp.token=''
```

3. **Set known token:**
```bash
docker run -e JUPYTER_TOKEN=mysecrettoken ...
```

### Kernel Keeps Dying

**Symptom:** Python kernel dies repeatedly

**Diagnosis:**
```bash
# Check memory
docker stats jupyter-username

# Check kernel logs
docker exec jupyter-username cat /home/jupyter/.local/share/jupyter/runtime/kernel-*.json
```

**Solutions:**

1. **Memory issue:** Increase container memory limit

2. **Package conflict:**
```bash
# Reset kernel environment
docker exec jupyter-username pip install --force-reinstall ipykernel
```

### Extensions Not Loading

**Symptom:** JupyterLab extensions missing

**Solution:**
```bash
# Rebuild extensions
docker exec jupyter-username jupyter lab build

# Or in Dockerfile
RUN jupyter lab build
```

---

## Docker Compose Issues

### Service Won't Start

**Diagnosis:**
```bash
# Check compose file syntax
docker-compose config

# Verbose output
docker-compose --verbose up
```

### Environment Variables Not Applied

**Diagnosis:**
```bash
# Check variable interpolation
docker-compose config

# Check .env file is in correct location
ls -la .env
```

**Solution:** Ensure `.env` file is in same directory as `docker-compose.yml`

### Volume Mount Empty

**Diagnosis:**
```bash
# Check path exists on host
ls -la /path/to/workspace

# Check volume in compose
docker-compose config | grep -A5 volumes
```

---

## Diagnostic Commands

### Quick Health Check Script

```bash
#!/bin/bash
# health-check.sh

echo "=== Host System ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
docker --version
docker-compose --version

echo ""
echo "=== Container Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "=== Resource Usage ==="
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

echo ""
echo "=== GPU in Containers ==="
for container in $(docker ps --format "{{.Names}}"); do
    echo "Container: $container"
    docker exec $container nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>/dev/null || echo "  No GPU access"
done
```

### Collect Debug Information

```bash
#!/bin/bash
# collect-debug.sh

OUTFILE="debug-$(date +%Y%m%d-%H%M%S).txt"

{
    echo "=== System Info ==="
    uname -a
    cat /etc/os-release
    
    echo ""
    echo "=== Docker Info ==="
    docker info
    docker version
    
    echo ""
    echo "=== NVIDIA Info ==="
    nvidia-smi
    cat /proc/driver/nvidia/version
    
    echo ""
    echo "=== Container Logs ==="
    for container in $(docker ps -a --format "{{.Names}}"); do
        echo "--- $container ---"
        docker logs --tail 50 $container 2>&1
    done
    
    echo ""
    echo "=== Docker Networks ==="
    docker network ls
    docker network inspect bridge
    
} > $OUTFILE

echo "Debug info saved to $OUTFILE"
```

---

## Getting Help

If this guide doesn't resolve your issue:

1. **Collect debug info:** Run `./scripts/collect-debug.sh`
2. **Check container logs:** `docker logs <container-name>`
3. **Contact support** with:
   - Debug output file
   - Steps to reproduce
   - Expected vs actual behavior
