# Container Management - Ansible Edition

## Overview

Deploy GPU-enabled Docker containers to VMs using Ansible with PKI authentication.

**User specifies:**
- **Project name** → Container name + workspace directory
- **Target VM** → Where to deploy

**System uses:**
- **User's UID** → Container runs as you
- **User's PKI cert** → SSH authentication (no passwords)

```
User: ./deploy.sh my-ml-project gpu-vm-01

Result:
  - Container: my-ml-project (running as your UID)
  - Workspace: /home/<you>/projects/my-ml-project/
  - Access: http://gpu-vm-01:10100
```

---

## Quick Start

### 1. Add Your VMs

Edit `inventory/hosts.yml`:
```yaml
gpu_vms:
  hosts:
    gpu-vm-01:
      ansible_host: 10.0.1.101
    gpu-vm-02:
      ansible_host: 10.0.1.102
```

### 2. Setup VM (one-time, requires sudo)

```bash
./deploy.sh setup gpu-vm-01
```

### 3. Deploy Your Project

```bash
./deploy.sh my-ml-project gpu-vm-01
```

### 4. Access Jupyter

```
URL printed at end of deployment
Your workspace: ~/projects/my-ml-project/workspace/
```

### 5. Stop/Remove When Done

```bash
./deploy.sh stop my-ml-project gpu-vm-01
./deploy.sh remove my-ml-project gpu-vm-01
```

---

## Commands

| Command | Description |
|---------|-------------|
| `./deploy.sh <project> <vm>` | Deploy container |
| `./deploy.sh stop <project> <vm>` | Stop container |
| `./deploy.sh start <project> <vm>` | Restart container |
| `./deploy.sh remove <project> <vm>` | Remove container |
| `./deploy.sh list <vm>` | List your containers |
| `./deploy.sh setup <vm>` | Setup VM (admin) |

---

## How It Works

```
┌─────────────────┐
│  Your Machine   │
│                 │
│  ./deploy.sh    │
│  my-project     │
│  gpu-vm-01      │
└────────┬────────┘
         │
         │ SSH with your PKI cert
         │ Ansible playbook runs
         ▼
┌─────────────────┐
│   gpu-vm-01     │
│                 │
│  Creates:       │
│  ~/projects/    │
│    my-project/  │
│      workspace/ │
│                 │
│  Runs:          │
│  Container      │
│  "my-project"   │
│  as YOUR UID    │
│  with GPU       │
└─────────────────┘
```

---

## Directory Structure

```
container-management-ansible/
├── README.md
├── ansible.cfg              # PKI/SSH configuration
├── deploy.sh                # Main CLI script
│
├── inventory/
│   ├── hosts.yml           # Your VMs
│   └── group_vars/
│       └── all.yml         # Default settings
│
├── playbooks/
│   ├── deploy.yml          # Deploy container
│   ├── stop.yml            # Stop container
│   ├── start.yml           # Start container
│   ├── remove.yml          # Remove container
│   ├── list.yml            # List containers
│   └── setup-vm.yml        # Setup VM (admin)
│
├── templates/
│   └── docker-compose.yml.j2
│
└── files/
    └── gpu-jupyter/
        ├── Dockerfile
        └── requirements.txt
```

---

## PKI Configuration

Your PKI certificate is used for SSH. Edit `ansible.cfg`:

```ini
[ssh_connection]
# Standard PKI key
ssh_args = -i ~/.ssh/your-pki-key

# OR for CAC/PIV smart card
ssh_args = -o PKCS11Provider=/usr/lib/opensc-pkcs11.so
```

---

## Port Assignment

Each user gets a unique port range based on UID:

```
UID 1000 → Ports 10000-10099
UID 1001 → Ports 10100-10199
UID 1002 → Ports 10200-10299
```

First project uses first port in range, second uses second, etc.
