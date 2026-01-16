#!/bin/bash
# =============================================================================
# CONTAINER MANAGEMENT SYSTEM - ONE-CLICK DEPLOYMENT
# =============================================================================
#
# Deploys GPU-enabled container management portal to any VM
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/yj07538Johnny/Johnny_AI_ML_LLM_experiments_2024/main/container-management/scripts/deploy.sh | bash
#
# Or locally:
#   ./deploy.sh
#
# Options:
#   --skip-build    Skip Docker image build (if image already exists)
#   --skip-portal   Skip Streamlit portal setup
#   --registry URL  Use custom registry instead of building locally
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/yj07538Johnny/Johnny_AI_ML_LLM_experiments_2024.git"
INSTALL_DIR="${HOME}/container-management"
IMAGE_NAME="ghcr.io/your-org/gpu-jupyter:latest"
PORTAL_PORT=8501

# Parse arguments
SKIP_BUILD=false
SKIP_PORTAL=false
CUSTOM_REGISTRY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-portal)
            SKIP_PORTAL=true
            shift
            ;;
        --registry)
            CUSTOM_REGISTRY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 found"
        return 0
    else
        log_error "$1 not found"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# BANNER
# -----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}       GPU CONTAINER MANAGEMENT SYSTEM - DEPLOYMENT SCRIPT                  ${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# -----------------------------------------------------------------------------
# STEP 1: CHECK PREREQUISITES
# -----------------------------------------------------------------------------

log_info "Checking prerequisites..."

PREREQ_FAILED=false

# Check Docker
if ! check_command docker; then
    log_error "Docker is required. Install: https://docs.docker.com/engine/install/"
    PREREQ_FAILED=true
fi

# Check NVIDIA drivers
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log_success "NVIDIA drivers found ($GPU_COUNT GPUs detected)"
else
    log_warn "nvidia-smi not found - GPU support may not work"
fi

# Check NVIDIA Container Toolkit
if docker info 2>/dev/null | grep -q "nvidia"; then
    log_success "NVIDIA Container Toolkit configured"
elif command -v nvidia-ctk &> /dev/null; then
    log_success "nvidia-ctk found"
else
    log_warn "NVIDIA Container Toolkit may not be configured"
    log_warn "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Check Python
if check_command python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $PYTHON_VERSION"
else
    log_error "Python 3 is required"
    PREREQ_FAILED=true
fi

# Check pip
if ! check_command pip3 && ! check_command pip; then
    log_error "pip is required"
    PREREQ_FAILED=true
fi

# Check git
if ! check_command git; then
    log_error "git is required"
    PREREQ_FAILED=true
fi

if [ "$PREREQ_FAILED" = true ]; then
    log_error "Prerequisites check failed. Please install missing components."
    exit 1
fi

echo ""
log_success "All prerequisites satisfied"
echo ""

# -----------------------------------------------------------------------------
# STEP 2: CLONE/UPDATE REPOSITORY
# -----------------------------------------------------------------------------

log_info "Setting up repository..."

if [ -d "$INSTALL_DIR" ]; then
    log_info "Directory exists, updating..."
    cd "$INSTALL_DIR"
    git pull origin main || log_warn "Could not pull updates"
else
    log_info "Cloning repository..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Navigate to container-management
cd container-management 2>/dev/null || cd Johnny_AI_ML_LLM_experiments_2024/container-management 2>/dev/null || true

log_success "Repository ready at: $(pwd)"
echo ""

# -----------------------------------------------------------------------------
# STEP 3: BUILD DOCKER IMAGE
# -----------------------------------------------------------------------------

if [ "$SKIP_BUILD" = false ]; then
    if [ -n "$CUSTOM_REGISTRY" ]; then
        log_info "Pulling image from custom registry: $CUSTOM_REGISTRY"
        docker pull "$CUSTOM_REGISTRY"
        docker tag "$CUSTOM_REGISTRY" "$IMAGE_NAME"
    else
        log_info "Building Docker image (this may take 5-10 minutes)..."
        echo ""

        # Check if image already exists
        if docker images "$IMAGE_NAME" --format "{{.Repository}}" | grep -q "gpu-jupyter"; then
            log_warn "Image already exists. Rebuilding..."
        fi

        # Build the image
        cd images/gpu-jupyter
        docker build -t "$IMAGE_NAME" .
        cd ../..

        log_success "Docker image built: $IMAGE_NAME"
    fi
else
    log_info "Skipping Docker build (--skip-build specified)"
fi

# Verify image exists
if docker images "$IMAGE_NAME" --format "{{.Repository}}" | grep -q "gpu-jupyter"; then
    IMAGE_SIZE=$(docker images "$IMAGE_NAME" --format "{{.Size}}")
    log_success "Image ready: $IMAGE_NAME ($IMAGE_SIZE)"
else
    log_error "Image not found: $IMAGE_NAME"
    exit 1
fi

echo ""

# -----------------------------------------------------------------------------
# STEP 4: SETUP STREAMLIT PORTAL
# -----------------------------------------------------------------------------

if [ "$SKIP_PORTAL" = false ]; then
    log_info "Setting up Streamlit portal..."

    cd streamlit-service 2>/dev/null || cd ../streamlit-service 2>/dev/null

    # Install Python dependencies
    log_info "Installing Python dependencies..."
    pip3 install --quiet streamlit docker python-dotenv pyyaml 2>/dev/null || \
    pip install --quiet streamlit docker python-dotenv pyyaml

    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log_info "Creating .env configuration..."
        cat > .env << 'EOF'
# Container Deployment Service Configuration
VM_HOST=localhost
REGISTRY=ghcr.io/your-org

# Development mode - remove in production
DEV_USER=admin
DEV_ADMIN=true

# Resource defaults
DEFAULT_MEMORY_LIMIT=32g
DEFAULT_SHM_SIZE=2g
USER_PORT_START=10000
EOF
        log_success "Created .env file"
    fi

    log_success "Streamlit portal ready"
else
    log_info "Skipping portal setup (--skip-portal specified)"
fi

echo ""

# -----------------------------------------------------------------------------
# STEP 5: SUMMARY & INSTRUCTIONS
# -----------------------------------------------------------------------------

echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN}                    DEPLOYMENT COMPLETE!                                    ${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""
echo -e "${BLUE}Image:${NC}  $IMAGE_NAME"
echo -e "${BLUE}Portal:${NC} $(pwd)"
echo ""
echo -e "${YELLOW}TO START THE PORTAL:${NC}"
echo ""
echo "  cd $(pwd)"
echo "  streamlit run app.py --server.port $PORTAL_PORT"
echo ""
echo -e "${YELLOW}TO DEPLOY A CONTAINER MANUALLY:${NC}"
echo ""
echo "  docker run -d \\"
echo "    --name gpu-jupyter-\$(whoami) \\"
echo "    --gpus all \\"
echo "    --memory 32g \\"
echo "    -p 10000:8888 \\"
echo "    -p 10001:6006 \\"
echo "    -v \$HOME/workspace:/workspace \\"
echo "    $IMAGE_NAME"
echo ""
echo -e "${YELLOW}TO ACCESS JUPYTER:${NC}"
echo ""
echo "  1. Get token: docker exec gpu-jupyter-\$(whoami) jupyter server list"
echo "  2. Open: http://localhost:10000?token=<TOKEN>"
echo ""
echo -e "${YELLOW}TO ENTER CONTAINER SHELL:${NC}"
echo ""
echo "  docker exec -it gpu-jupyter-\$(whoami) /bin/bash"
echo ""
echo -e "${GREEN}=============================================================================${NC}"

# -----------------------------------------------------------------------------
# OPTIONAL: START PORTAL NOW
# -----------------------------------------------------------------------------

echo ""
read -p "Start the portal now? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Starting Streamlit portal on port $PORTAL_PORT..."
    echo ""
    echo "Access at: http://localhost:$PORTAL_PORT"
    echo "Press Ctrl+C to stop"
    echo ""
    streamlit run app.py --server.port $PORTAL_PORT --server.headless true
fi
