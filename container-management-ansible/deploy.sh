#!/bin/bash
# =============================================================================
# GPU CONTAINER DEPLOYMENT CLI
# =============================================================================
#
# Usage:
#   ./deploy.sh <project-name> <target-vm>          Deploy container
#   ./deploy.sh stop <project-name> <target-vm>     Stop container
#   ./deploy.sh start <project-name> <target-vm>   Start container
#   ./deploy.sh remove <project-name> <target-vm>  Remove container
#   ./deploy.sh list <target-vm>                    List containers
#   ./deploy.sh setup <target-vm>                   Setup VM (admin)
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
usage() {
    echo -e "${CYAN}GPU Container Deployment${NC}"
    echo ""
    echo "Usage:"
    echo "  ./deploy.sh <project-name> <target-vm>    Deploy new container"
    echo "  ./deploy.sh stop <project-name> <vm>      Stop container"
    echo "  ./deploy.sh start <project-name> <vm>     Start container"  
    echo "  ./deploy.sh remove <project-name> <vm>    Remove container"
    echo "  ./deploy.sh list <vm>                     List your containers"
    echo "  ./deploy.sh setup <vm>                    Setup VM (requires sudo)"
    echo ""
    echo "Options:"
    echo "  --image <name>     Image to use (default: gpu-jupyter)"
    echo "  --memory <size>    Memory limit (default: 32g)"
    echo "  --gpus <count>     GPU count: all, 1, 2 (default: all)"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh sentiment-analysis gpu-vm-01"
    echo "  ./deploy.sh my-training gpu-vm-02 --memory 64g --gpus 2"
    echo "  ./deploy.sh stop my-training gpu-vm-02"
    exit 1
}

# -----------------------------------------------------------------------------
# PARSE ARGUMENTS
# -----------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    usage
fi

ACTION=""
PROJECT_NAME=""
TARGET_VM=""
IMAGE="gpu-jupyter"
MEMORY="32g"
GPUS="all"

# First argument determines action
case "$1" in
    setup|stop|start|remove|list)
        ACTION="$1"
        shift
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        # Default action is deploy
        ACTION="deploy"
        ;;
esac

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
        *)
            if [ -z "$PROJECT_NAME" ]; then
                PROJECT_NAME="$1"
            elif [ -z "$TARGET_VM" ]; then
                TARGET_VM="$1"
            fi
            shift
            ;;
    esac
done

# -----------------------------------------------------------------------------
# VALIDATE
# -----------------------------------------------------------------------------
validate_project_name() {
    if [[ ! "$1" =~ ^[a-zA-Z][a-zA-Z0-9_-]*$ ]]; then
        echo -e "${RED}Error: Project name must start with letter, contain only letters/numbers/hyphens/underscores${NC}"
        exit 1
    fi
}

# Get user info (for all actions)
USER_NAME=$(whoami)
USER_UID=$(id -u)
USER_GID=$(id -g)

# -----------------------------------------------------------------------------
# ACTIONS
# -----------------------------------------------------------------------------

do_deploy() {
    if [ -z "$PROJECT_NAME" ] || [ -z "$TARGET_VM" ]; then
        echo -e "${RED}Error: deploy requires <project-name> <target-vm>${NC}"
        usage
    fi
    
    validate_project_name "$PROJECT_NAME"
    
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Deploying Container${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Project:    $PROJECT_NAME"
    echo "  Target VM:  $TARGET_VM"
    echo "  User:       $USER_NAME (UID: $USER_UID)"
    echo "  Image:      $IMAGE"
    echo "  Memory:     $MEMORY"
    echo "  GPUs:       $GPUS"
    echo ""
    
    ansible-playbook playbooks/deploy.yml \
        -l "$TARGET_VM" \
        -e "project_name=$PROJECT_NAME" \
        -e "user_name=$USER_NAME" \
        -e "user_uid=$USER_UID" \
        -e "user_gid=$USER_GID" \
        -e "container_image=$IMAGE" \
        -e "memory_limit=$MEMORY" \
        -e "gpu_count=$GPUS"
}

do_stop() {
    if [ -z "$PROJECT_NAME" ] || [ -z "$TARGET_VM" ]; then
        echo -e "${RED}Error: stop requires <project-name> <target-vm>${NC}"
        usage
    fi
    
    echo -e "${YELLOW}Stopping $PROJECT_NAME on $TARGET_VM...${NC}"
    
    ansible-playbook playbooks/stop.yml \
        -l "$TARGET_VM" \
        -e "project_name=$PROJECT_NAME" \
        -e "user_name=$USER_NAME"
}

do_start() {
    if [ -z "$PROJECT_NAME" ] || [ -z "$TARGET_VM" ]; then
        echo -e "${RED}Error: start requires <project-name> <target-vm>${NC}"
        usage
    fi
    
    echo -e "${GREEN}Starting $PROJECT_NAME on $TARGET_VM...${NC}"
    
    ansible-playbook playbooks/start.yml \
        -l "$TARGET_VM" \
        -e "project_name=$PROJECT_NAME" \
        -e "user_name=$USER_NAME"
}

do_remove() {
    if [ -z "$PROJECT_NAME" ] || [ -z "$TARGET_VM" ]; then
        echo -e "${RED}Error: remove requires <project-name> <target-vm>${NC}"
        usage
    fi
    
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  WARNING: This will remove container and workspace data${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Project: $PROJECT_NAME"
    echo "  VM:      $TARGET_VM"
    echo ""
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi
    
    ansible-playbook playbooks/remove.yml \
        -l "$TARGET_VM" \
        -e "project_name=$PROJECT_NAME" \
        -e "user_name=$USER_NAME"
}

do_list() {
    if [ -z "$PROJECT_NAME" ]; then
        # First positional arg is actually the VM for list command
        TARGET_VM="$PROJECT_NAME"
        PROJECT_NAME=""
    fi
    
    if [ -z "$TARGET_VM" ]; then
        echo -e "${RED}Error: list requires <target-vm>${NC}"
        usage
    fi
    
    echo -e "${CYAN}Listing containers on $TARGET_VM for user $USER_NAME...${NC}"
    echo ""
    
    ansible-playbook playbooks/list.yml \
        -l "$TARGET_VM" \
        -e "user_name=$USER_NAME"
}

do_setup() {
    # For setup, first arg is the VM
    TARGET_VM="$PROJECT_NAME"
    
    if [ -z "$TARGET_VM" ]; then
        echo -e "${RED}Error: setup requires <target-vm>${NC}"
        usage
    fi
    
    echo -e "${CYAN}Setting up $TARGET_VM (requires sudo on target)...${NC}"
    echo ""
    
    ansible-playbook playbooks/setup-vm.yml \
        -l "$TARGET_VM" \
        --become \
        --ask-become-pass
}

# -----------------------------------------------------------------------------
# EXECUTE
# -----------------------------------------------------------------------------
case "$ACTION" in
    deploy)
        do_deploy
        ;;
    stop)
        do_stop
        ;;
    start)
        do_start
        ;;
    remove)
        do_remove
        ;;
    list)
        do_list
        ;;
    setup)
        do_setup
        ;;
    *)
        usage
        ;;
esac
