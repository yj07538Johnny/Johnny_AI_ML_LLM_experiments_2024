# Enterprise GPU Container Management System

## Architecture Overview

This system provides curated, GPU-enabled Docker containers to data scientists through a self-service web interface with PKI authentication.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   REGISTRY LAYER              SERVICE LAYER              COMPUTE LAYER   │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐  │
│  │                 │      │                 │      │                 │  │
│  │  Curated Image  │      │   Deployment    │      │    User VM      │  │
│  │    Registry     │─────►│    Service      │─────►│   + GPU         │  │
│  │   (ghcr.io/     │      │  (Streamlit)    │      │   Container     │  │
│  │    internal)    │      │                 │      │                 │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘  │
│          │                        │                        │             │
│          │                   PKI Auth                 nvidia-docker      │
│          │                   via CAC                   --gpus all        │
│          ▼                        ▼                        ▼             │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐  │
│  │  - NVIDIA base  │      │  - User creds   │      │  - Port 8888    │  │
│  │  - Approved pkgs│      │  - VM mapping   │      │    (Jupyter)    │  │
│  │  - Pinned vers  │      │  - Audit logs   │      │  - Port 6006    │  │
│  │  - Scan results │      │  - Quotas       │      │    (TensorBoard)│  │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Curated Image Registry

**Purpose:** Store approved, scanned, versioned container images.

**Location:** 
- Development: `ghcr.io/your-org/`
- Production: Internal ContainerYard

**Images Provided:**

| Image Name | Base | Purpose | GPU |
|------------|------|---------|-----|
| `gpu-jupyter` | nvidia/cuda:12.2.0-runtime | Interactive development | Yes |
| `gpu-pytorch` | nvcr.io/nvidia/pytorch | PyTorch training | Yes |
| `gpu-tensorflow` | nvcr.io/nvidia/tensorflow | TensorFlow training | Yes |
| `ml-cpu` | python:3.11-slim | CPU-only analysis | No |

### 2. Deployment Service (Streamlit)

**Purpose:** Web interface for data scientists to deploy containers.

**Features:**
- PKI/CAC authentication
- Image selection from curated list
- Resource allocation (GPU count, memory)
- Port assignment (isolated per user)
- Container lifecycle management (start/stop/restart)
- Log viewing

### 3. User VM + Container

**Purpose:** Isolated compute environment with GPU access.

**Configuration:**
- One VM per user (or shared pool)
- NVIDIA Container Toolkit installed
- Port ranges assigned per user
- Network isolation enforced

---

## Security Model

### Network Isolation

```
EXTERNAL NETWORK                    INTERNAL NETWORK
(User workstations)                 (Container services)
       │                                   │
       │ :443 (HTTPS)                      │
       ▼                                   │
┌─────────────────┐                        │
│  Load Balancer  │                        │
│  / Reverse Proxy│                        │
└────────┬────────┘                        │
         │                                 │
         │ PKI Validated                   │
         ▼                                 │
┌─────────────────┐      ┌─────────────────┐
│   Streamlit     │      │   Container     │
│   Deployment    │─────►│   Registry      │
│   Service       │      │   (Internal)    │
└────────┬────────┘      └─────────────────┘
         │
         │ SSH/Docker API
         ▼
┌─────────────────┐
│   User VM       │
│  ┌───────────┐  │
│  │ Container │  │◄── GPU Passthrough
│  │ (Jupyter) │  │
│  └───────────┘  │
│   Ports:        │
│   8888 (Jupyter)│◄── Mapped externally per user
│   6006 (TB)     │
└─────────────────┘
```

### Port Assignment Strategy

Each user gets an isolated port range:

```
User A: 10000-10099
User B: 10100-10199
User C: 10200-10299
...
```

Mapping:
- `10000` → Container's `8888` (Jupyter)
- `10001` → Container's `6006` (TensorBoard)
- `10002` → Container's `22` (SSH if enabled)

### Authentication Flow

```
1. User accesses Streamlit UI
2. PKI certificate presented (CAC/PIV)
3. Certificate validated against CA
4. User identity extracted from cert CN
5. User mapped to VM assignment
6. Deployment authorized for that VM only
```

---

## File Structure

```
container-management/
├── README.md                    # This file
├── docs/
│   ├── ADMIN_GUIDE.md          # Administration procedures
│   ├── TROUBLESHOOTING.md      # Common issues and fixes
│   ├── SECURITY.md             # Security architecture
│   └── USER_GUIDE.md           # End-user documentation
├── images/
│   ├── gpu-jupyter/
│   │   ├── Dockerfile          # Annotated Dockerfile
│   │   ├── requirements.txt    # Pinned Python packages
│   │   └── README.md           # Image-specific docs
│   ├── gpu-pytorch/
│   │   ├── Dockerfile
│   │   └── ...
│   └── gpu-tensorflow/
│       └── ...
├── deployment/
│   ├── docker-compose.yml      # Annotated compose file
│   ├── docker-compose.prod.yml # Production overrides
│   └── .env.example            # Environment template
├── streamlit-service/
│   ├── app.py                  # Main Streamlit application
│   ├── auth.py                 # PKI authentication
│   ├── docker_manager.py       # Container orchestration
│   └── requirements.txt
└── scripts/
    ├── setup-vm.sh             # VM preparation script
    ├── install-nvidia.sh       # NVIDIA toolkit installation
    └── health-check.sh         # System health verification
```

---

## Quick Start

### For Administrators

```bash
# 1. Clone repository
git clone https://github.com/your-org/container-management.git
cd container-management

# 2. Build curated images
cd images/gpu-jupyter
docker build -t ghcr.io/your-org/gpu-jupyter:latest .
docker push ghcr.io/your-org/gpu-jupyter:latest

# 3. Deploy Streamlit service
cd ../../streamlit-service
docker-compose up -d

# 4. Prepare user VMs
cd ../scripts
./setup-vm.sh <vm-hostname>
```

### For Data Scientists

1. Navigate to: `https://container-portal.your-org.com`
2. Authenticate with CAC/PIV
3. Select image (e.g., `gpu-jupyter`)
4. Click "Deploy"
5. Access Jupyter at provided URL

---

## Related Documentation

- [Administration Guide](docs/ADMIN_GUIDE.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [Security Architecture](docs/SECURITY.md)
- [User Guide](docs/USER_GUIDE.md)
- [Dockerfile Reference](images/gpu-jupyter/README.md)
- [Docker Compose Reference](deployment/README.md)
