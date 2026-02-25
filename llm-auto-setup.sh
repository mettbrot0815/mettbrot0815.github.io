#!/usr/bin/env bash
# =============================================================================
# Local LLM Auto-Setup — Universal Edition
# Scans your hardware and automatically selects the best model.
# No Hugging Face token required — all models are from public repos.
# Supports: Ubuntu 22.04 / 24.04 — CPU-only through high-end GPU.
# =============================================================================

set -uo pipefail

# ---------- Configuration -----------------------------------------------------
LOG_FILE="$HOME/llm-auto-setup-$(date +%Y%m%d-%H%M%S).log"
VENV_DIR="$HOME/.local/share/llm-venv"
MODEL_BASE="$HOME/local-llm-models"
OLLAMA_MODELS="$MODEL_BASE/ollama"
GGUF_MODELS="$MODEL_BASE/gguf"
TEMP_DIR="$MODEL_BASE/temp"
BIN_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.config/local-llm"
ALIAS_FILE="$HOME/.local_llm_aliases"
MODEL_CONFIG="$CONFIG_DIR/selected_model.conf"
GUI_DIR="$HOME/.local/share/llm-webui"

# ---------- Colors ------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; MAGENTA='\033[0;35m'; NC='\033[0m'

# ---------- Logging -----------------------------------------------------------
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

log()   { echo -e "$(date +'%Y-%m-%d %H:%M:%S') $1"; }
info()  { log "${GREEN}[INFO]${NC}  $1"; }
warn()  { log "${YELLOW}[WARN]${NC}  $1"; }
error() { log "${RED}[ERROR]${NC} $1"; exit 1; }
step()  {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  ▶  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}
highlight() {
    echo -e "\n${MAGENTA}  ◆  $1${NC}"
}

ask_yes_no() {
    local prompt="$1" ans=""
    if [[ ! -t 0 ]]; then warn "Non-interactive — treating '$prompt' as No."; return 1; fi
    read -r -p "$(echo -e "${YELLOW}?${NC} $prompt (y/N) ")" -n 1 ans; echo
    [[ "$ans" =~ ^[Yy]$ ]] && return 0 || return 1
}

retry() {
    local n="$1" delay="$2"; shift 2
    local attempt=1
    while true; do
        "$@" && return 0
        (( attempt >= n )) && { warn "Failed after $n attempts: $*"; return 1; }
        warn "Attempt $attempt/$n failed — retrying in ${delay}s…"
        sleep "$delay"; attempt=$(( attempt + 1 ))
    done
}

is_wsl2() {
    # Check /proc/version for "microsoft" (present on all WSL1 and WSL2 kernels).
    # We intentionally do NOT require "wsl2" in uname -r — most WSL2 kernel strings
    # contain "microsoft-standard" or "microsoft-WSL2" but not the literal "wsl2".
    grep -qi microsoft /proc/version 2>/dev/null
}

# =============================================================================
# STEP 1 — PRE-FLIGHT
# =============================================================================
step "Pre-flight checks"

[[ "${EUID}" -eq 0 ]] && error "Do not run as root."
command -v sudo &>/dev/null || error "sudo is required."

# ── Single sudo prompt — keep credentials alive for the entire script ─────────
# User types their password exactly once here. A background loop refreshes the
# sudo timestamp every 50 s so it never expires mid-install.
echo -e "${CYAN}[sudo]${NC} This script needs elevated privileges for apt, systemd, and CUDA/ROCm."
sudo -v || error "sudo authentication failed."
( while true; do sudo -n true; sleep 50; done ) &
SUDO_KEEPALIVE_PID=$!
# Ensure keepalive is killed even if script exits early (error, Ctrl-C, etc.)
trap 'kill "$SUDO_KEEPALIVE_PID" 2>/dev/null' EXIT INT TERM
info "sudo keepalive active (PID $SUDO_KEEPALIVE_PID)."

info "Log: $LOG_FILE"
is_wsl2 && info "WSL2 detected." || info "Native Linux detected."

# =============================================================================
# STEP 2 — SYSTEM SCAN
# =============================================================================
step "Hardware detection"

# ---------- CPU ---------------------------------------------------------------
CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "Unknown")
CPU_THREADS=$(nproc 2>/dev/null || echo 4)

# Detect instruction sets (important for choosing optimal llama.cpp build)
CPU_FLAGS=$(grep -m1 'flags' /proc/cpuinfo 2>/dev/null || echo "")
HAS_AVX2=0;   echo "$CPU_FLAGS" | grep -qw avx2   && HAS_AVX2=1
HAS_AVX512=0; echo "$CPU_FLAGS" | grep -qw avx512f && HAS_AVX512=1
HAS_AVX=0;    echo "$CPU_FLAGS" | grep -qw avx     && HAS_AVX=1

# ---------- RAM ---------------------------------------------------------------
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 4096000)
AVAIL_RAM_KB=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 2048000)
TOTAL_RAM_GB=$(( TOTAL_RAM_KB / 1024 / 1024 ))
AVAIL_RAM_GB=$(( AVAIL_RAM_KB / 1024 / 1024 ))
# Ensure sane minimums in case of weird /proc/meminfo output
(( TOTAL_RAM_GB < 1 )) && TOTAL_RAM_GB=4
(( AVAIL_RAM_GB < 1 )) && AVAIL_RAM_GB=2

# ---------- GPU ---------------------------------------------------------------
# We detect NVIDIA and AMD independently, then set unified HAS_GPU / GPU_VRAM_GB
# so the model selection engine works identically for both.
HAS_NVIDIA=0
HAS_AMD_GPU=0
HAS_GPU=0          # set to 1 if any capable GPU found
GPU_NAME="None"
GPU_VRAM_MIB=0
GPU_VRAM_GB=0
DRIVER_VER="N/A"
CUDA_VER_SMI=""
AMD_ROCM_VER=""

# ── NVIDIA ────────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "Unknown")
    GPU_VRAM_MIB_RAW=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo "0")
    [[ "$GPU_VRAM_MIB_RAW" =~ ^[0-9]+$ ]] && GPU_VRAM_MIB=$GPU_VRAM_MIB_RAW || GPU_VRAM_MIB=0
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "N/A")
    CUDA_VER_SMI=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -n1 || echo "")
    if (( GPU_VRAM_MIB > 500 )); then
        HAS_NVIDIA=1; HAS_GPU=1
        GPU_VRAM_GB=$(( GPU_VRAM_MIB / 1024 ))
    fi
fi

# ── AMD GPU ───────────────────────────────────────────────────────────────────
# Only probe AMD if no NVIDIA found — avoid dual-GPU confusion.
if (( !HAS_NVIDIA )); then
    # sysfs mem_info_vram_total works on all kernels ≥ 4.15 without ROCm installed.
    # Iterate cards; pick the first one reporting > 512 MiB (skips iGPUs).
    for _sysfs_card in /sys/class/drm/card*/device/mem_info_vram_total; do
        [[ -f "$_sysfs_card" ]] || continue
        _amd_vram_bytes=$(cat "$_sysfs_card" 2>/dev/null || echo 0)
        _amd_vram_mib=$(( _amd_vram_bytes / 1024 / 1024 ))
        if (( _amd_vram_mib > 512 )); then
            GPU_VRAM_MIB=$_amd_vram_mib
            GPU_VRAM_GB=$(( _amd_vram_mib / 1024 ))
            HAS_AMD_GPU=1; HAS_GPU=1
            # Get GPU name from lspci (works without ROCm)
            GPU_NAME=$(lspci 2>/dev/null                 | grep -i "VGA\|Display\|3D"                 | grep -i "AMD\|ATI\|Radeon\|gfx"                 | head -n1 | sed 's/.*: //' | xargs || echo "AMD GPU")
            # ROCm version if installed
            if command -v rocminfo &>/dev/null; then
                AMD_ROCM_VER=$(rocminfo 2>/dev/null | grep -oP 'Runtime Version: \K[0-9.]+' | head -n1 || echo "")
            fi
            DRIVER_VER=$(cat /sys/class/drm/card*/device/driver/module/version 2>/dev/null | head -n1                 || echo "$(uname -r)")
            break
        fi
    done
fi

# ---------- Disk --------------------------------------------------------------
DISK_FREE_GB=$(df -BG "$HOME" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}' || echo 10)

# ---------- Print summary -----------------------------------------------------
echo ""
echo -e "  ${CYAN}┌─────────────────────────────────────────────┐${NC}"
echo -e "  ${CYAN}│           HARDWARE SCAN RESULTS             │${NC}"
echo -e "  ${CYAN}├─────────────────────────────────────────────┤${NC}"
printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "CPU" "$CPU_MODEL" | cut -c1-52
printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "Threads"  "${CPU_THREADS} logical cores"
printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "SIMD"     "$(
    flags=""
    (( HAS_AVX512 )) && flags="AVX-512 AVX2 AVX"
    [[ -z "$flags" ]] && (( HAS_AVX2 )) && flags="AVX2 AVX"
    [[ -z "$flags" ]] && (( HAS_AVX  )) && flags="AVX"
    [[ -z "$flags" ]] && flags="baseline (no AVX)"
    echo "$flags"
)"
printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "RAM"      "${TOTAL_RAM_GB} GB total / ${AVAIL_RAM_GB} GB free"
printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "GPU"      "$GPU_NAME"
if (( HAS_NVIDIA )); then
    printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "VRAM"     "${GPU_VRAM_GB} GB (${GPU_VRAM_MIB} MiB)"
    printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "Driver"   "$DRIVER_VER"
    [[ -n "$CUDA_VER_SMI" ]] && \
    printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "API"      "CUDA $CUDA_VER_SMI"
elif (( HAS_AMD_GPU )); then
    printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "VRAM"     "${GPU_VRAM_GB} GB (${GPU_VRAM_MIB} MiB)"
    printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "Driver"   "$DRIVER_VER"
    [[ -n "$AMD_ROCM_VER" ]] && \
    printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "API"      "ROCm $AMD_ROCM_VER" || \
    printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "API"      "ROCm (not yet installed)"
fi
printf "  ${CYAN}│${NC}  %-12s %-30s ${CYAN}│${NC}\n" "Disk free" "${DISK_FREE_GB} GB"
echo -e "  ${CYAN}└─────────────────────────────────────────────┘${NC}"
echo ""

# =============================================================================
# STEP 3 — MODEL SELECTION ENGINE
# =============================================================================
# All models are public (no HF token). All from bartowski or TheBloke.
#
# Decision tree:
#   VRAM >= 24 GB → 32B / Mixtral
#   VRAM >= 16 GB → 22B Q4
#   VRAM >= 12 GB → 14B Q4 / Nemo 12B Q5
#   VRAM >=  8 GB → 8B Q6 / Nemo 12B Q4
#   VRAM >=  6 GB → 8B Q6 (primary), 8B Q4 (fallback)
#   VRAM >=  4 GB → 8B Q4 / 7B Q4
#   VRAM >=  2 GB → 3B Q6 / 7B Q2
#   No GPU / <2GB → 3B Q4 CPU-only
#
# CPU layers (n_offload): fills remaining VRAM, rest to RAM.
# We cap CPU layers so RAM usage stays under (TOTAL_RAM_GB - 4) GB.
# =============================================================================

step "Auto-selecting model"

# Helper: MiB needed per layer for a given model total size and layer count
# Used to calculate how many layers fit in VRAM
# $1=model_size_gb  $2=num_layers  → MiB per layer
mib_per_layer() {
    local size_gb="$1" layers="$2"
    echo $(( (size_gb * 1024) / layers ))
}

# Headroom to keep free in VRAM for KV-cache + activations
VRAM_HEADROOM_MIB=1400
VRAM_USABLE_MIB=$(( GPU_VRAM_MIB - VRAM_HEADROOM_MIB ))
(( VRAM_USABLE_MIB < 0 )) && VRAM_USABLE_MIB=0

# Maximum RAM we want to commit to model layers (leave 4 GB for OS + Python)
RAM_FOR_LAYERS_GB=$(( TOTAL_RAM_GB - 4 ))
(( RAM_FOR_LAYERS_GB < 1 )) && RAM_FOR_LAYERS_GB=1

# Calculate layers that fit fully in VRAM given model size and layer count
# Returns the number of layers to offload to GPU
gpu_layers_for() {
    local size_gb="$1" num_layers="$2"
    local mib_layer=$(( (size_gb * 1024) / num_layers ))
    (( mib_layer < 1 )) && mib_layer=1
    local layers=$(( VRAM_USABLE_MIB / mib_layer ))
    (( layers > num_layers )) && layers=$num_layers
    (( layers < 0 )) && layers=0
    echo $layers
}

# ── Model definitions ─────────────────────────────────────────────────────────
# Capability tags shown in the picker:
#   [TOOLS]  = structured tool/function calling (agents, APIs, JSON output)
#   [THINK]  = thinking/reasoning mode via <think> tokens
#              Use: add "/think" to prompt, or set temp 0.6 + top_p 0.95
#              Skip: add "/no_think" for fast plain answers
#   [UNCENS] = uncensored fine-tune — no content restrictions
#   ★        = best recommended pick for that VRAM tier
#
# All models from bartowski public repos — no HF token needed.

declare -A M   # holds the chosen model's fields

select_model() {
    local vram=$GPU_VRAM_GB
    local ram=$TOTAL_RAM_GB

    # ── ≥ 24 GB VRAM ─────────────────────────────────────────────────────────
    if (( HAS_GPU && vram >= 24 )); then
        highlight "High-end GPU (${vram} GB VRAM) → Qwen2.5-32B [TOOLS] ★"
        M[name]="Qwen2.5-32B-Instruct Q4_K_M"; M[caps]="TOOLS"
        M[file]="Qwen2.5-32B-Instruct-Q4_K_M.gguf"
        M[url]="https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
        M[size_gb]=19; M[layers]=64; M[tier]="32B"; return

    # ── ≥ 16 GB VRAM ─────────────────────────────────────────────────────────
    elif (( HAS_GPU && vram >= 16 )); then
        highlight "16 GB VRAM → Qwen3-30B-A3B MoE [TOOLS+THINK] ★"
        M[name]="Qwen3-30B-A3B Q4_K_M (MoE)"; M[caps]="TOOLS + THINK"
        M[file]="Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"
        M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/resolve/main/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"
        M[size_gb]=18; M[layers]=48; M[tier]="30B-A3B (MoE)"; return

    # ── ≥ 12 GB VRAM ─────────────────────────────────────────────────────────
    elif (( HAS_GPU && vram >= 12 )); then
        highlight "12 GB VRAM → Qwen3-14B [TOOLS+THINK] ★"
        M[name]="Qwen3-14B Q4_K_M"; M[caps]="TOOLS + THINK"
        M[file]="Qwen_Qwen3-14B-Q4_K_M.gguf"
        M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF/resolve/main/Qwen_Qwen3-14B-Q4_K_M.gguf"
        M[size_gb]=9; M[layers]=40; M[tier]="14B"; return

    # ── ≥ 10 GB VRAM ─────────────────────────────────────────────────────────
    elif (( HAS_GPU && vram >= 10 )); then
        highlight "10 GB VRAM → Mistral-Nemo-12B [TOOLS]"
        M[name]="Mistral-Nemo-12B Q5_K_M"; M[caps]="TOOLS"
        M[file]="Mistral-Nemo-Instruct-2407-Q5_K_M.gguf"
        M[url]="https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q5_K_M.gguf"
        M[size_gb]=8; M[layers]=40; M[tier]="12B"; return

    # ── ≥ 8 GB VRAM ──────────────────────────────────────────────────────────
    elif (( HAS_GPU && vram >= 8 )); then
        highlight "8 GB VRAM → Qwen3-8B Q6 [TOOLS+THINK] ★"
        M[name]="Qwen3-8B Q6_K"; M[caps]="TOOLS + THINK"
        M[file]="Qwen_Qwen3-8B-Q6_K.gguf"
        M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q6_K.gguf"
        M[size_gb]=6; M[layers]=36; M[tier]="8B"; return

    # ── ≥ 6 GB VRAM ──────────────────────────────────────────────────────────
    elif (( HAS_GPU && vram >= 6 )); then
        highlight "6 GB VRAM → Qwen3-8B Q4 [TOOLS+THINK] ★"
        M[name]="Qwen3-8B Q4_K_M"; M[caps]="TOOLS + THINK"
        M[file]="Qwen_Qwen3-8B-Q4_K_M.gguf"
        M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q4_K_M.gguf"
        M[size_gb]=5; M[layers]=36; M[tier]="8B"; return

    # ── ≥ 4 GB VRAM ──────────────────────────────────────────────────────────
    elif (( HAS_GPU && vram >= 4 )); then
        highlight "4 GB VRAM → Qwen3-4B Q4 [TOOLS+THINK]"
        M[name]="Qwen3-4B Q4_K_M"; M[caps]="TOOLS + THINK"
        M[file]="Qwen_Qwen3-4B-Q4_K_M.gguf"
        M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf"
        M[size_gb]=3; M[layers]=36; M[tier]="4B"; return

    # ── ≥ 2 GB VRAM (partial offload) ────────────────────────────────────────
    elif (( HAS_GPU && vram >= 2 )); then
        highlight "Small GPU (${vram} GB) → Phi-3.5-mini partial offload"
        M[name]="Phi-3.5-mini-instruct Q4_K_M"; M[caps]="none"
        M[file]="Phi-3.5-mini-instruct-Q4_K_M.gguf"
        M[url]="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
        M[size_gb]=2; M[layers]=32; M[tier]="3.8B"; return

    # ── CPU-only ──────────────────────────────────────────────────────────────
    else
        if (( ram >= 16 )); then
            highlight "CPU-only (${ram} GB RAM) → Qwen3-8B Q4 [TOOLS+THINK] ★"
            M[name]="Qwen3-8B Q4_K_M"; M[caps]="TOOLS + THINK"
            M[file]="Qwen_Qwen3-8B-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q4_K_M.gguf"
            M[size_gb]=5; M[layers]=36; M[tier]="8B"
        elif (( ram >= 8 )); then
            highlight "CPU-only (${ram} GB RAM) → Qwen3-4B Q4 [TOOLS+THINK]"
            M[name]="Qwen3-4B Q4_K_M"; M[caps]="TOOLS + THINK"
            M[file]="Qwen_Qwen3-4B-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf"
            M[size_gb]=3; M[layers]=36; M[tier]="4B"
        else
            highlight "Low RAM CPU-only → Phi-3.5-mini Q4 (most efficient)"
            M[name]="Phi-3.5-mini-instruct Q4_K_M"; M[caps]="none"
            M[file]="Phi-3.5-mini-instruct-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
            M[size_gb]=2; M[layers]=32; M[tier]="3.8B"
        fi
        return
    fi
}

select_model

# Calculate optimal GPU and CPU layer counts
if (( HAS_GPU )); then
    GPU_LAYERS=$(gpu_layers_for "${M[size_gb]}" "${M[layers]}")
    CPU_LAYERS=$(( M[layers] - GPU_LAYERS ))
    (( CPU_LAYERS < 0 )) && CPU_LAYERS=0
else
    GPU_LAYERS=0
    CPU_LAYERS="${M[layers]}"
fi

# Clamp CPU layers to available RAM
MIB_PER_LAYER=$(( (M[size_gb] * 1024) / M[layers] ))
MAX_CPU_LAYERS=$(( (RAM_FOR_LAYERS_GB * 1024) / (MIB_PER_LAYER > 0 ? MIB_PER_LAYER : 1) ))
(( CPU_LAYERS > MAX_CPU_LAYERS )) && CPU_LAYERS=$MAX_CPU_LAYERS

# Optimal thread count (physical cores, capped at 16)
# Detect physical (non-hyperthreaded) core count for optimal inference threading.
# We run lscpu once and parse both fields to avoid spawning two subshells.
LSCPU_OUT=$(lscpu 2>/dev/null || true)
PHYS_ONLY=$(echo "$LSCPU_OUT" | awk '/^Core\(s\) per socket/{print $NF}')
SOCKETS=$(echo   "$LSCPU_OUT" | awk '/^Socket\(s\)/{print $NF}')
if [[ -n "$PHYS_ONLY" && -n "$SOCKETS" && "$PHYS_ONLY" =~ ^[0-9]+$ && "$SOCKETS" =~ ^[0-9]+$ ]]; then
    HW_THREADS=$(( PHYS_ONLY * SOCKETS ))
else
    # Fall back to logical core count from /proc/cpuinfo
    HW_THREADS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 4)
fi
(( HW_THREADS > 16 )) && HW_THREADS=16

# Batch size: scale with VRAM
if (( GPU_VRAM_GB >= 16 )); then  BATCH=1024
elif (( GPU_VRAM_GB >= 8 ));  then BATCH=512
elif (( GPU_VRAM_GB >= 4 ));  then BATCH=256
else                               BATCH=128
fi

# Print recommendation box
VRAM_USED_GB=$(( (GPU_LAYERS * MIB_PER_LAYER) / 1024 ))
RAM_USED_GB=$(( (CPU_LAYERS  * MIB_PER_LAYER) / 1024 ))

echo ""
echo -e "  ${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "  ${GREEN}║           RECOMMENDED CONFIGURATION                 ║${NC}"
echo -e "  ${GREEN}╠══════════════════════════════════════════════════════╣${NC}"
printf "  ${GREEN}║${NC}  %-16s %-35s${GREEN}║${NC}\n" "Model"         "${M[name]}"
printf "  ${GREEN}║${NC}  %-16s %-35s${GREEN}║${NC}\n" "Capabilities"  "${M[caps]}"
printf "  ${GREEN}║${NC}  %-16s %-35s${GREEN}║${NC}\n" "Size"          "${M[tier]}  (~${M[size_gb]} GB file)"
printf "  ${GREEN}║${NC}  %-16s %-35s${GREEN}║${NC}\n" "GPU layers"    "${GPU_LAYERS} / ${M[layers]}  (~${VRAM_USED_GB} GB VRAM)"
printf "  ${GREEN}║${NC}  %-16s %-35s${GREEN}║${NC}\n" "CPU layers"    "${CPU_LAYERS}  (~${RAM_USED_GB} GB RAM)"
printf "  ${GREEN}║${NC}  %-16s %-35s${GREEN}║${NC}\n" "Threads"       "${HW_THREADS}"
printf "  ${GREEN}║${NC}  %-16s %-35s${GREEN}║${NC}\n" "Batch size"    "${BATCH}"
echo -e "  ${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

if ! ask_yes_no "Proceed with this configuration?"; then
    echo ""
    echo -e "  ${CYAN}━━━━━━━━━━━━━━━━━━  MODEL PICKER  ━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  Capability legend:"
    echo -e "    ${GREEN}[TOOLS]${NC}   tool/function calling — agents, JSON, APIs"
    echo -e "    ${YELLOW}[THINK]${NC}   thinking mode — add /think to prompt for step-by-step reasoning"
    echo -e "                             add /no_think for fast plain answers"
    echo -e "    ${MAGENTA}[UNCENS]${NC}  uncensored — no content restrictions (fine-tuned)"
    echo -e "    ${CYAN}★${NC}         best pick for that VRAM tier"
    echo ""
    echo "  ┌────┬──────────────────────────────┬──────┬──────┬──────────────────────────┐"
    echo "  │ #  │ Model                        │ Quant│ VRAM │ Capabilities             │"
    echo "  ├────┼──────────────────────────────┼──────┼──────┼──────────────────────────┤"
    echo "  │  1 │ Phi-3.5-mini 3.8B            │ Q4   │ CPU  │ (basic chat)             │"
    echo "  │  2 │ Qwen3-1.7B                   │ Q8   │ CPU  │ [TOOLS] [THINK]          │"
    echo "  │  3 │ Qwen3-4B                     │ Q4   │ ~3GB │ ★ [TOOLS] [THINK]        │"
    echo "  │  4 │ Qwen2.5-3B                   │ Q6   │ ~2GB │ [TOOLS]                  │"
    echo "  │  5 │ Qwen3-8B                     │ Q4   │ ~5GB │ ★ [TOOLS] [THINK]        │"
    echo "  │  6 │ Qwen3-8B                     │ Q6   │ ~6GB │ ★ [TOOLS] [THINK]        │"
    echo "  │  7 │ Dolphin3.0-8B                │ Q4   │ ~5GB │ [UNCENS]                 │"
    echo "  │  8 │ Dolphin3.0-8B                │ Q6   │ ~6GB │ [UNCENS]                 │"
    echo "  │  9 │ Mistral-Nemo-12B             │ Q4   │ ~7GB │ [TOOLS]                  │"
    echo "  │ 10 │ Mistral-Nemo-12B             │ Q5   │ ~8GB │ [TOOLS]                  │"
    echo "  │ 11 │ Qwen3-14B                    │ Q4   │ ~9GB │ ★ [TOOLS] [THINK]        │"
    echo "  │ 12 │ Qwen2.5-14B                  │ Q4   │ ~9GB │ [TOOLS]                  │"
    echo "  │ 13 │ Mistral-Small-22B            │ Q4   │~13GB │ [TOOLS]                  │"
    echo "  │ 14 │ Qwen3-30B-A3B  (MoE ★fast)  │ Q4   │~16GB │ ★ [TOOLS] [THINK]        │"
    echo "  │ 15 │ Qwen2.5-32B                  │ Q4   │~19GB │ [TOOLS]                  │"
    echo "  └────┴──────────────────────────────┴──────┴──────┴──────────────────────────┘"
    echo ""
    echo -e "  ${YELLOW}MoE note (14):${NC} 30B params, only 3B active per token → 30B quality at 8B speed."
    echo ""
    read -r -p "  Choice [1-15]: " manual_choice
    case "$manual_choice" in
        1)  M[name]="Phi-3.5-mini-instruct Q4_K_M";     M[caps]="none"
            M[file]="Phi-3.5-mini-instruct-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
            M[size_gb]=2;  M[layers]=32; M[tier]="3.8B" ;;
        2)  M[name]="Qwen3-1.7B Q8_0";                  M[caps]="TOOLS + THINK"
            M[file]="Qwen_Qwen3-1.7B-Q8_0.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q8_0.gguf"
            M[size_gb]=2;  M[layers]=28; M[tier]="1.7B" ;;
        3)  M[name]="Qwen3-4B Q4_K_M";                  M[caps]="TOOLS + THINK"
            M[file]="Qwen_Qwen3-4B-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf"
            M[size_gb]=3;  M[layers]=36; M[tier]="4B" ;;
        4)  M[name]="Qwen2.5-3B-Instruct Q6_K";         M[caps]="TOOLS"
            M[file]="Qwen2.5-3B-Instruct-Q6_K.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q6_K.gguf"
            M[size_gb]=2;  M[layers]=36; M[tier]="3B" ;;
        5)  M[name]="Qwen3-8B Q4_K_M";                  M[caps]="TOOLS + THINK"
            M[file]="Qwen_Qwen3-8B-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q4_K_M.gguf"
            M[size_gb]=5;  M[layers]=36; M[tier]="8B" ;;
        6)  M[name]="Qwen3-8B Q6_K";                    M[caps]="TOOLS + THINK"
            M[file]="Qwen_Qwen3-8B-Q6_K.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q6_K.gguf"
            M[size_gb]=6;  M[layers]=36; M[tier]="8B" ;;
        7)  M[name]="Dolphin3.0-Llama3.1-8B Q4_K_M";   M[caps]="UNCENS"
            M[file]="Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Dolphin3.0-Llama3.1-8B-GGUF/resolve/main/Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf"
            M[size_gb]=5;  M[layers]=32; M[tier]="8B" ;;
        8)  M[name]="Dolphin3.0-Llama3.1-8B Q6_K";     M[caps]="UNCENS"
            M[file]="Dolphin3.0-Llama3.1-8B-Q6_K.gguf"
            M[url]="https://huggingface.co/bartowski/Dolphin3.0-Llama3.1-8B-GGUF/resolve/main/Dolphin3.0-Llama3.1-8B-Q6_K.gguf"
            M[size_gb]=6;  M[layers]=32; M[tier]="8B" ;;
        9)  M[name]="Mistral-Nemo-12B Q4_K_M";          M[caps]="TOOLS"
            M[file]="Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
            M[size_gb]=7;  M[layers]=40; M[tier]="12B" ;;
        10) M[name]="Mistral-Nemo-12B Q5_K_M";          M[caps]="TOOLS"
            M[file]="Mistral-Nemo-Instruct-2407-Q5_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q5_K_M.gguf"
            M[size_gb]=8;  M[layers]=40; M[tier]="12B" ;;
        11) M[name]="Qwen3-14B Q4_K_M";                 M[caps]="TOOLS + THINK"
            M[file]="Qwen_Qwen3-14B-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF/resolve/main/Qwen_Qwen3-14B-Q4_K_M.gguf"
            M[size_gb]=9;  M[layers]=40; M[tier]="14B" ;;
        12) M[name]="Qwen2.5-14B-Instruct Q4_K_M";      M[caps]="TOOLS"
            M[file]="Qwen2.5-14B-Instruct-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
            M[size_gb]=9;  M[layers]=48; M[tier]="14B" ;;
        13) M[name]="Mistral-Small-22B Q4_K_M";         M[caps]="TOOLS"
            M[file]="Mistral-Small-22B-ArliAI-RPMax-v1.1-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Mistral-Small-22B-ArliAI-RPMax-v1.1-GGUF/resolve/main/Mistral-Small-22B-ArliAI-RPMax-v1.1-Q4_K_M.gguf"
            M[size_gb]=13; M[layers]=48; M[tier]="22B" ;;
        14) M[name]="Qwen3-30B-A3B Q4_K_M (MoE)";      M[caps]="TOOLS + THINK"
            M[file]="Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/resolve/main/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"
            M[size_gb]=18; M[layers]=48; M[tier]="30B-A3B" ;;
        15) M[name]="Qwen2.5-32B-Instruct Q4_K_M";      M[caps]="TOOLS"
            M[file]="Qwen2.5-32B-Instruct-Q4_K_M.gguf"
            M[url]="https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
            M[size_gb]=19; M[layers]=64; M[tier]="32B" ;;
        *)  warn "Invalid choice — keeping auto-selected model." ;;
    esac

    # Recalculate layers and batch for manually chosen model
    if (( HAS_GPU )); then
        GPU_LAYERS=$(gpu_layers_for "${M[size_gb]}" "${M[layers]}")
        CPU_LAYERS=$(( M[layers] - GPU_LAYERS ))
        (( CPU_LAYERS < 0 )) && CPU_LAYERS=0
    else
        GPU_LAYERS=0; CPU_LAYERS="${M[layers]}"
    fi
    if (( GPU_VRAM_GB >= 16 )); then  BATCH=1024
    elif (( GPU_VRAM_GB >= 8 ));  then BATCH=512
    elif (( GPU_VRAM_GB >= 4 ));  then BATCH=256
    else                               BATCH=128
    fi
fi  # end manual override block
MODEL_SIZE_GB="${M[size_gb]}"
if (( DISK_FREE_GB < MODEL_SIZE_GB + 2 )); then
    warn "Low disk space: ${DISK_FREE_GB} GB free, model needs ~${MODEL_SIZE_GB} GB."
    ask_yes_no "Continue anyway?" || error "Aborting — free up disk space and re-run."
fi

# =============================================================================
# STEP 3b — PYTHON INSTALLATION
# =============================================================================
step "Python environment"

# Ensure TEMP_DIR exists — used for venv test and get-pip.py bootstrap
mkdir -p "$TEMP_DIR"

# ── apt-get update first — this step needs packages before anything else ──────
info "Running apt-get update…"
sudo apt-get update -qq || warn "apt update returned non-zero."

# ── Detect current Python version ─────────────────────────────────────────────
PYVER_RAW=$(python3 --version 2>/dev/null | grep -oP '\d+\.\d+' | head -n1 || echo "0.0")
PYVER_MAJOR=$(echo "$PYVER_RAW" | cut -d. -f1)
PYVER_MINOR=$(echo "$PYVER_RAW" | cut -d. -f2)
info "System Python: ${PYVER_RAW:-not found}"

PYTHON_BIN="python3"

# ── If Python < 3.10: install 3.11 via deadsnakes PPA ─────────────────────────
# llama-cpp-python pre-built wheels exist for 3.10/3.11/3.12.
# Ubuntu 20.04 ships 3.8 and needs an upgrade. 22.04→3.10, 24.04→3.12 are fine.
if (( PYVER_MAJOR < 3 || (PYVER_MAJOR == 3 && PYVER_MINOR < 10) )); then
    warn "Python $PYVER_RAW is too old (need 3.10+). Installing Python 3.11 via deadsnakes PPA…"
    sudo apt-get install -y software-properties-common 2>/dev/null || true
    if ! grep -rq "deadsnakes" /etc/apt/sources.list.d/ 2>/dev/null; then
        sudo add-apt-repository -y ppa:deadsnakes/ppa \
            || warn "Failed to add deadsnakes PPA — will try system python."
        sudo apt-get update -qq || true
    fi
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev \
        || warn "python3.11 install failed — falling back to system python."
    command -v python3.11 &>/dev/null && PYTHON_BIN="python3.11" \
        && info "Using Python 3.11 for venv."
else
    info "Python $PYVER_RAW ✔ — meets 3.10+ requirement."
fi

# ── Refresh PYVER_* from the actual binary we will use ────────────────────────
# If we just installed python3.11 via deadsnakes, PYVER_MAJOR/MINOR still hold
# the old system python version (e.g. 3.8). The venv package install below uses
# these variables, so we must update them to match PYTHON_BIN.
_PYVER_REFRESH=$("$PYTHON_BIN" --version 2>/dev/null | grep -oP '\d+\.\d+' | head -n1 || echo "$PYVER_RAW")
PYVER_MAJOR=$(echo "$_PYVER_REFRESH" | cut -d. -f1)
PYVER_MINOR=$(echo "$_PYVER_REFRESH" | cut -d. -f2)
unset _PYVER_REFRESH

# ── Install pip + venv for the detected version ───────────────────────────────
# On Ubuntu 24.04, python3-venv alone is not enough — python3.12-venv is needed.
# We install both the generic and version-specific packages to cover all cases.
info "Installing python3-pip, python3-venv, python${PYVER_MAJOR}.${PYVER_MINOR}-venv…"
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    "python${PYVER_MAJOR}.${PYVER_MINOR}-venv" \
    "python${PYVER_MAJOR}.${PYVER_MINOR}-dev" \
    2>/dev/null \
    || warn "Some Python packages failed — will attempt to continue."

# If pip still not available, bootstrap it
if ! "$PYTHON_BIN" -m pip --version &>/dev/null 2>&1; then
    info "pip not found — bootstrapping via get-pip.py…"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$TEMP_DIR/get-pip.py" \
        && "$PYTHON_BIN" "$TEMP_DIR/get-pip.py" --quiet \
        && rm -f "$TEMP_DIR/get-pip.py" \
        || warn "get-pip.py bootstrap failed — pip may be unavailable."
fi

# Upgrade pip to latest
"$PYTHON_BIN" -m pip install --upgrade pip --quiet 2>/dev/null \
    || warn "pip upgrade failed — using whatever version is installed."
PIP_VER=$("$PYTHON_BIN" -m pip --version 2>/dev/null | awk '{print $2}' || echo "unknown")
info "pip $PIP_VER ✔"

# ── Verify venv works before proceeding ───────────────────────────────────────
TEST_VENV="$TEMP_DIR/.test_venv_$$"
if "$PYTHON_BIN" -m venv "$TEST_VENV" 2>/dev/null; then
    rm -rf "$TEST_VENV"
    info "Python venv: OK  ($("$PYTHON_BIN" --version 2>&1))"
else
    # Last resort: try to install the venv module directly
    warn "venv test failed — trying to install python3-venv one more time…"
    sudo apt-get install -y "python${PYVER_MAJOR}.${PYVER_MINOR}-venv" python3-venv 2>/dev/null || true
    if ! "$PYTHON_BIN" -m venv "$TEST_VENV" 2>/dev/null; then
        error "Python venv creation still failing. Run manually: sudo apt-get install python${PYVER_MAJOR}.${PYVER_MINOR}-venv"
    fi
    rm -rf "$TEST_VENV"
    info "Python venv: OK after reinstall."
fi

export PYTHON_BIN

# =============================================================================
# STEP 4 — SYSTEM DEPENDENCIES
# =============================================================================
step "System dependencies"

# Note: apt-get update already ran in the Python environment step above

PKGS=(curl wget git build-essential cmake ninja-build python3 lsb-release zstd ffmpeg pciutils)
(( HAS_AVX2 )) && PKGS+=(libopenblas-dev)   # AVX2 path for CPU layers

sudo apt-get install -y "${PKGS[@]}" || warn "Some packages may have failed."

for cmd in curl wget git python3; do
    command -v "$cmd" &>/dev/null || error "Critical dependency missing: $cmd"
done
# pip is accessed via python3 -m pip (no standalone pip3 on Ubuntu 24.04)
"$PYTHON_BIN" -m pip --version &>/dev/null || error "pip not available — check Python environment step above."
info "System dependencies OK."

# =============================================================================
# STEP 5 — DIRECTORIES + PATH
# =============================================================================
step "Directories"
mkdir -p "$OLLAMA_MODELS" "$GGUF_MODELS" "$TEMP_DIR" "$BIN_DIR" "$CONFIG_DIR" "$GUI_DIR"
info "Directories ready."

for _rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [[ -f "$_rc" ]] && ! grep -q "# llm-auto-setup PATH" "$_rc"; then
        { echo ""; echo "# llm-auto-setup PATH"
          echo "[[ \":\$PATH:\" != *\":$BIN_DIR:\"* ]] && export PATH=\"$BIN_DIR:\$PATH\""; } >> "$_rc"
        info "Added $BIN_DIR to PATH in $_rc"
    fi
done
[[ ":$PATH:" != *":$BIN_DIR:"* ]] && export PATH="$BIN_DIR:$PATH"

# =============================================================================
# STEP 6 — NVIDIA DRIVER CHECK
# =============================================================================
if (( HAS_NVIDIA )); then
    step "NVIDIA driver"
    info "GPU: $GPU_NAME | Driver: $DRIVER_VER | VRAM: ${GPU_VRAM_MIB} MiB"
elif (( HAS_AMD_GPU )); then
    step "AMD GPU detected"
    info "GPU: $GPU_NAME | VRAM: ${GPU_VRAM_MIB} MiB | Driver: $DRIVER_VER"
    [[ -n "$AMD_ROCM_VER" ]] && info "ROCm already present: $AMD_ROCM_VER"
else
    info "No discrete GPU found — running CPU-only mode."
fi

# =============================================================================
# STEP 7 — CUDA TOOLKIT (skip if no GPU)
# =============================================================================
if (( HAS_NVIDIA )); then
    step "CUDA toolkit"

    setup_cuda_env() {
        # Run ldconfig first so symlinks like libcudart.so.12 are created from
        # libcudart.so.12.x.y.z before we try to find them.
        sudo ldconfig 2>/dev/null || true

        local lib_dir=""
        # Search for libcudart.so.12* (wildcard catches .12, .12.x, .12.x.y.z)
        # Also check the standard CUDA targets path which find sometimes misses at low depth
        while IFS= read -r -d '' p; do lib_dir="$(dirname "$p")"; break
        done < <(find /usr/local /usr/lib /opt \
                    -maxdepth 8 \
                    \( -name "libcudart.so.12" -o -name "libcudart.so.12.*" \) \
                    -print0 2>/dev/null)

        # Fallback: check ldconfig cache directly
        if [[ -z "$lib_dir" ]]; then
            local ldcache_path
            ldcache_path=$(ldconfig -p 2>/dev/null | grep 'libcudart\.so\.12' | awk '{print $NF}' | head -n1 || true)
            [[ -n "$ldcache_path" ]] && lib_dir="$(dirname "$ldcache_path")"
        fi

        if [[ -z "$lib_dir" ]]; then
            warn "libcudart.so.12 not found in filesystem or ldconfig cache."
            warn "  This usually means CUDA installed but ldconfig hasn't run yet."
            warn "  Try: sudo ldconfig && source ~/.bashrc"
            return 1
        fi

        export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
        info "CUDA libs found at: $lib_dir"

        # Walk up to find the CUDA root (handles /usr/local/cuda-12.x/targets/arch/lib)
        local base_dir; base_dir="$(echo "$lib_dir" | sed 's|/targets/.*||; s|/lib[^/]*$||')"
        local bin_dir="$base_dir/bin"
        [[ -d "$bin_dir" ]] && { export PATH="$bin_dir:$PATH"; info "CUDA bin: $bin_dir"; }

        for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
            [[ -f "$rc" ]] && ! grep -q "# CUDA toolkit — llm-auto-setup" "$rc" && {
                { echo ""; echo "# CUDA toolkit — llm-auto-setup"
                  [[ -d "$bin_dir" ]] && echo "export PATH=\"${bin_dir}:\$PATH\""
                  echo "export LD_LIBRARY_PATH=\"${lib_dir}:\${LD_LIBRARY_PATH:-}\""; } >> "$rc"; }
        done
        return 0
    }

    # Three-probe CUDA detection (PATH → filesystem → ldconfig/dpkg)
    CUDA_PRESENT=0
    if ! command -v nvcc &>/dev/null; then
        NVCC_PATH=$(find /usr/local /usr/lib/cuda /opt/cuda -maxdepth 6 -name nvcc -type f 2>/dev/null | head -n1 || true)
        [[ -n "$NVCC_PATH" ]] && { export PATH="$(dirname "$NVCC_PATH"):$PATH"; info "nvcc at $NVCC_PATH"; }
    fi
    command -v nvcc &>/dev/null && CUDA_PRESENT=1
    if (( !CUDA_PRESENT )); then
        # Wildcard: catches libcudart.so.12, libcudart.so.12.x, libcudart.so.12.x.y.z
        find /usr/local /usr/lib /opt -maxdepth 8 \
            \( -name "libcudart.so.12" -o -name "libcudart.so.12.*" \) 2>/dev/null | grep -q . \
            && CUDA_PRESENT=1
    fi
    if (( !CUDA_PRESENT )); then
        ldconfig -p 2>/dev/null | grep -q 'libcudart\.so\.12' && CUDA_PRESENT=1
    fi
    if (( !CUDA_PRESENT )); then
        dpkg -l 'cuda-toolkit-*' 'cuda-libraries-*' 2>/dev/null | grep -q '^ii' && CUDA_PRESENT=1
    fi

    if (( CUDA_PRESENT )); then
        info "CUDA already installed: $(nvcc --version 2>/dev/null | grep release | head -n1 || echo 'present')"
        setup_cuda_env || true
    else
        info "Installing CUDA toolkit…"
        UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "unknown")
        if [[ "$UBUNTU_VERSION" != "22.04" && "$UBUNTU_VERSION" != "24.04" ]]; then
            warn "Ubuntu $UBUNTU_VERSION not tested. Attempting anyway."
        fi
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION//./}/x86_64/cuda-keyring_1.1-1_all.deb"
        retry 3 5 wget -q -O "$TEMP_DIR/cuda-keyring.deb" "$KEYRING_URL" \
            || error "Failed to download CUDA keyring."
        sudo dpkg -i "$TEMP_DIR/cuda-keyring.deb" || true
        rm -f "$TEMP_DIR/cuda-keyring.deb"
        sudo apt-get update -qq || true
        CUDA_PKG=$(apt-cache search --names-only '^cuda-toolkit-12-' 2>/dev/null | awk '{print $1}' | sort -V | tail -n1 || true)
        [[ -z "$CUDA_PKG" ]] && CUDA_PKG="cuda-toolkit"
        sudo apt-get install -y "$CUDA_PKG" || warn "CUDA install returned non-zero."
        info "Running ldconfig to register CUDA libraries…"
        sudo ldconfig 2>/dev/null || true
        setup_cuda_env || true
    fi

    ldconfig -p 2>/dev/null | grep -q "libcudart.so.12" && info "libcudart.so.12 in ldconfig ✔"
fi

# =============================================================================
# STEP 7b — ROCm TOOLKIT (AMD GPU only)
# =============================================================================
if (( HAS_AMD_GPU && !HAS_NVIDIA )); then
    step "ROCm toolkit (AMD GPU)"

    setup_rocm_env() {
        # Add ROCm lib path to LD_LIBRARY_PATH and persist it
        local rocm_lib=""
        for _rp in /opt/rocm/lib /opt/rocm-*/lib /usr/lib/x86_64-linux-gnu; do
            if [[ -f "$_rp/libhipblas.so" || -f "$_rp/librocblas.so" ]]; then
                rocm_lib="$_rp"; break
            fi
        done
        [[ -z "$rocm_lib" ]] && rocm_lib="/opt/rocm/lib"   # best guess
        export LD_LIBRARY_PATH="$rocm_lib:${LD_LIBRARY_PATH:-}"
        export PATH="/opt/rocm/bin:$PATH"
        for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
            [[ -f "$rc" ]] && ! grep -q "# ROCm — llm-auto-setup" "$rc" && {
                # printf with single-quoted format: $PATH stays literal (expands at shell startup).
                # $rocm_lib expands now (we want the real path baked in).
                printf '\n# ROCm — llm-auto-setup\n' >> "$rc"
                printf 'export PATH="/opt/rocm/bin:$PATH"\n' >> "$rc"
                printf 'export LD_LIBRARY_PATH="%s:${LD_LIBRARY_PATH:-}"\n' "$rocm_lib" >> "$rc"
            }
        done
        info "ROCm env configured: $rocm_lib"
    }

    ROCM_PRESENT=0
    command -v rocminfo &>/dev/null && ROCM_PRESENT=1
    [[ -d /opt/rocm ]] && ROCM_PRESENT=1

    if (( ROCM_PRESENT )); then
        info "ROCm already installed."
        AMD_ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null             || rocminfo 2>/dev/null | grep -oP 'Runtime Version: \K[0-9.]+' | head -n1             || echo "present")
        info "ROCm version: $AMD_ROCM_VER"
        setup_rocm_env
    else
        info "Installing ROCm via amdgpu-install…"
        UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "unknown")
        # amdgpu-install is the official AMD installer — handles kernel modules + ROCm
        # Discover the current deb filename dynamically from the /latest/ directory listing
        # so we don't hardcode a version that may 404 after AMD ships a new release.
        AMDGPU_BASE="https://repo.radeon.com/amdgpu-install/latest/ubuntu/${UBUNTU_VERSION}/"
        AMDGPU_DEB=$(wget -qO- "$AMDGPU_BASE" 2>/dev/null             | grep -oP 'amdgpu-install_[^"]+_all\.deb' | tail -1             || echo "amdgpu-install_6.3.60300-1_all.deb")
        AMDGPU_DEB_URL="${AMDGPU_BASE}${AMDGPU_DEB}"
        info "AMD installer: $AMDGPU_DEB"
        if retry 3 10 wget -q -O "$TEMP_DIR/amdgpu-install.deb" "$AMDGPU_DEB_URL"; then
            sudo dpkg -i "$TEMP_DIR/amdgpu-install.deb" || true
            sudo apt-get update -qq || true
            rm -f "$TEMP_DIR/amdgpu-install.deb"
            # rocm metapackage includes HIP, hipBLAS, rocBLAS — everything llama.cpp needs
            sudo amdgpu-install --usecase=rocm --no-dkms -y                 || warn "amdgpu-install returned non-zero — ROCm may be partially installed."
            setup_rocm_env
        else
            warn "Failed to download amdgpu-install deb — trying manual apt path…"
            # Fallback: direct apt install of minimal ROCm components
            wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key 2>/dev/null                 | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg || true
            echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.3 ${UBUNTU_VERSION} main"                 | sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null
            sudo apt-get update -qq || true
            sudo apt-get install -y rocm-hip-sdk rocm-opencl-sdk                 || warn "ROCm apt install failed — check https://rocm.docs.amd.com"
            setup_rocm_env
        fi
        # Add current user to render + video groups (required for GPU access)
        sudo usermod -aG render,video "$USER" 2>/dev/null             && info "Added $USER to render+video groups (takes effect on next login)." || true
    fi

    # Verify HIP is usable
    if command -v hipconfig &>/dev/null; then
        info "HIP: $(hipconfig --version 2>/dev/null || echo 'present') ✔"
    else
        warn "hipconfig not found — ROCm may need a reboot to fully activate."
    fi
fi

# =============================================================================
# STEP 8 — PYTHON VENV
# =============================================================================
step "Python virtual environment"

[[ ! -d "$VENV_DIR" ]] && "${PYTHON_BIN:-python3}" -m venv "$VENV_DIR" || true
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate" || error "Failed to activate venv."
[[ "${VIRTUAL_ENV:-}" != "$VENV_DIR" ]] && error "Venv activation failed."
info "Venv: $VIRTUAL_ENV"
pip install --upgrade pip setuptools wheel --quiet || true

# =============================================================================
# STEP 9 — LLAMA-CPP-PYTHON
# =============================================================================
step "llama-cpp-python"

check_python_module() { "$VENV_DIR/bin/python3" -c "import $1" 2>/dev/null; }

# Build flags tuned to detected CPU features
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
(( HAS_NVIDIA ))  && CMAKE_ARGS+=" -DGGML_CUDA=ON -DLLAMA_CUBLAS=ON"
(( HAS_AVX512 ))  && CMAKE_ARGS+=" -DGGML_AVX512=ON -DGGML_AVX2=ON -DGGML_FMA=ON"
(( !HAS_AVX512 && HAS_AVX2 )) && CMAKE_ARGS+=" -DGGML_AVX2=ON -DGGML_FMA=ON"
(( !HAS_AVX2 && HAS_AVX )) && CMAKE_ARGS+=" -DGGML_AVX=ON"
export SOURCE_BUILD_CMAKE_ARGS="$CMAKE_ARGS"

LLAMA_INSTALLED=0

# ── NVIDIA CUDA wheels ────────────────────────────────────────────────────────
if (( HAS_NVIDIA )); then
    CUDA_VER=""
    CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -n1 || true)
    [[ -z "$CUDA_VER" ]] && CUDA_VER="$CUDA_VER_SMI"
    [[ -z "$CUDA_VER" ]] && CUDA_VER="12.1"
    CUDA_TAG="cu$(echo "$CUDA_VER" | tr -d '.')"
    info "CUDA $CUDA_VER → wheel tag $CUDA_TAG"
    for wheel_url in \
        "https://abetlen.github.io/llama-cpp-python/whl/${CUDA_TAG}" \
        "https://abetlen.github.io/llama-cpp-python/whl/cu124" \
        "https://abetlen.github.io/llama-cpp-python/whl/cu122" \
        "https://abetlen.github.io/llama-cpp-python/whl/cu121"; do
        info "Trying CUDA wheel: $wheel_url"
        pip install llama-cpp-python \
            --index-url "$wheel_url" \
            --extra-index-url https://pypi.org/simple \
            --quiet 2>&1 && { info "CUDA wheel installed from $wheel_url"; LLAMA_INSTALLED=1; break; } \
            || warn "Failed — trying next."
    done
fi

# ── AMD ROCm/HIP wheels ───────────────────────────────────────────────────────
if (( HAS_AMD_GPU && !HAS_NVIDIA && LLAMA_INSTALLED == 0 )); then
    info "Trying ROCm pre-built wheels for llama-cpp-python…"
    for wheel_url in \
        "https://abetlen.github.io/llama-cpp-python/whl/rocm600" \
        "https://abetlen.github.io/llama-cpp-python/whl/rocm550"; do
        info "Trying ROCm wheel: $wheel_url"
        pip install llama-cpp-python \
            --index-url "$wheel_url" \
            --extra-index-url https://pypi.org/simple \
            --quiet 2>&1 && { info "ROCm wheel installed from $wheel_url"; LLAMA_INSTALLED=1; break; } \
            || warn "Failed — trying next."
    done
fi

# ── Source build fallback ─────────────────────────────────────────────────────
if (( LLAMA_INSTALLED == 0 )); then
    if (( HAS_NVIDIA )); then
        warn "No pre-built CUDA wheel found — building from source (~5 min)…"
        MAKE_JOBS="$HW_THREADS" CMAKE_ARGS="$SOURCE_BUILD_CMAKE_ARGS" \
        pip install llama-cpp-python --no-cache-dir \
            || warn "llama-cpp-python CUDA build failed. Check logs."
    elif (( HAS_AMD_GPU )); then
        warn "No pre-built ROCm wheel found — building from source (~8 min)…"
        # GGML_HIPBLAS=ON enables ROCm GPU offload in llama.cpp
        MAKE_JOBS="$HW_THREADS" \
        CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DGGML_HIPBLAS=ON" \
        pip install llama-cpp-python --no-cache-dir \
            || warn "llama-cpp-python ROCm build failed. Check logs."
    else
        info "CPU-only build — compiling llama-cpp-python (~3 min)…"
        MAKE_JOBS="$HW_THREADS" CMAKE_ARGS="$SOURCE_BUILD_CMAKE_ARGS" \
        pip install llama-cpp-python --no-cache-dir \
            || warn "llama-cpp-python CPU build failed. Check logs."
    fi
fi

if check_python_module llama_cpp; then info "llama-cpp-python ✔"
else warn "llama-cpp-python import failed — check CUDA paths."; fi

# =============================================================================
# STEP 10 — OLLAMA
# =============================================================================
step "Ollama"

if ! command -v ollama &>/dev/null; then
    info "Installing Ollama…"
    retry 3 10 bash -c "curl -fsSL https://ollama.com/install.sh | sh" </dev/null \
        || error "Ollama install failed."
else
    info "Ollama: $(ollama --version 2>/dev/null || echo 'already installed')"
fi

# Tune Ollama concurrency to detected RAM
OLLAMA_PARALLEL=1
(( TOTAL_RAM_GB >= 32 )) && OLLAMA_PARALLEL=2

if is_wsl2; then
    cat > "$BIN_DIR/ollama-start" <<OLSTART
#!/usr/bin/env bash
export OLLAMA_MODELS="$OLLAMA_MODELS"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_NUM_PARALLEL=$OLLAMA_PARALLEL
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_THREAD=$HW_THREADS
export OLLAMA_ORIGINS="*"
# ── GPU maximisation ──────────────────────────────────────────────────────────
# Flash attention halves KV-cache VRAM usage → more room for model layers on GPU
export OLLAMA_FLASH_ATTENTION=1
# Quantise the KV-cache to q8_0 — further reduces VRAM with negligible quality loss
# (q4_0 saves even more if you run very long contexts)
export OLLAMA_KV_CACHE_TYPE=q8_0
# ── AMD ROCm GPU vars (no-op on NVIDIA systems) ───────────────────────────────
# HSA_OVERRIDE_GFX_VERSION: needed for some RDNA2/3 cards not yet in ROCm whitelist
# ROCR_VISIBLE_DEVICES=0:  use first AMD GPU (safe default for single-GPU setups)
# Only set HSA_OVERRIDE_GFX_VERSION if user pre-defined it (e.g. for RX 6000/7000 whitelist bypass)
# Exporting it empty causes ROCm to treat it as unrecognised, which is worse than absent.
[[ -n "\${HSA_OVERRIDE_GFX_VERSION:-}" ]] && export HSA_OVERRIDE_GFX_VERSION
export ROCR_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES:-0}
pgrep -f "ollama serve" >/dev/null 2>&1 && { echo "Ollama already running."; exit 0; }
echo "Starting Ollama…"
nohup ollama serve >"\$HOME/.ollama.log" 2>&1 &
sleep 3
pgrep -f "ollama serve" >/dev/null 2>&1 && echo "Ollama started." || { echo "ERROR: Ollama failed to start. Check: cat ~/.ollama.log"; exit 1; }
OLSTART
    chmod +x "$BIN_DIR/ollama-start"
    "$BIN_DIR/ollama-start" || warn "Ollama launcher returned non-zero."
else
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<EOF
[Service]
Environment="OLLAMA_MODELS=$OLLAMA_MODELS"
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NUM_PARALLEL=$OLLAMA_PARALLEL"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_THREAD=$HW_THREADS"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="ROCR_VISIBLE_DEVICES=0"
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable ollama  || warn "systemctl enable ollama failed."
    sudo systemctl restart ollama || warn "systemctl restart ollama failed."

    # Write ollama-start wrapper for native Linux too (used by llm-chat launcher)
    cat > "$BIN_DIR/ollama-start" <<OLSTART_NATIVE
#!/usr/bin/env bash
if systemctl is-active --quiet ollama 2>/dev/null; then
    echo "Ollama service already running."
else
    echo "Starting Ollama service…"
    sudo systemctl start ollama || { echo "ERROR: sudo systemctl start ollama failed."; exit 1; }
    sleep 2
    systemctl is-active --quiet ollama && echo "Ollama started." || echo "WARNING: check: sudo journalctl -u ollama -n 30"
fi
OLSTART_NATIVE
    chmod +x "$BIN_DIR/ollama-start"
fi

sleep 3
if is_wsl2; then
    pgrep -f "ollama serve" >/dev/null 2>&1 && info "Ollama running." || warn "Ollama not running."
else
    systemctl is-active --quiet ollama && info "Ollama service active." || warn "Ollama service not active."
fi

# =============================================================================
# STEP 11 — SAVE CONFIG + DOWNLOAD MODEL
# =============================================================================
step "Model download"

cat > "$MODEL_CONFIG" <<EOF
MODEL_NAME="${M[name]}"
MODEL_URL="${M[url]}"
MODEL_FILENAME="${M[file]}"
MODEL_SIZE="${M[tier]}"
MODEL_CAPS="${M[caps]}"
MODEL_LAYERS="${M[layers]}"
GPU_LAYERS="$GPU_LAYERS"
CPU_LAYERS="$CPU_LAYERS"
HW_THREADS="$HW_THREADS"
BATCH="$BATCH"
EOF
info "Config saved: $MODEL_CONFIG"

if ask_yes_no "Download ${M[name]} (~${M[size_gb]} GB) now?"; then
    info "Downloading ${M[file]} → $GGUF_MODELS"
    pushd "$GGUF_MODELS" >/dev/null

    DL_OK=0
    if command -v curl &>/dev/null; then
        retry 3 20 curl -L --fail -C - --progress-bar \
            -o "${M[file]}" "${M[url]}" \
            && DL_OK=1 || warn "curl download failed."
    fi
    if [[ "$DL_OK" -eq 0 ]] && command -v wget &>/dev/null; then
        retry 3 20 wget --tries=1 --show-progress -c \
            -O "${M[file]}" "${M[url]}" \
            && DL_OK=1 || warn "wget download also failed."
    fi

    if [[ "$DL_OK" -eq 1 && -f "${M[file]}" ]]; then
        info "Download complete: $(du -h "${M[file]}" | cut -f1)"

        # ── Register GGUF with Ollama ─────────────────────────────────────────
        # This makes the model appear in Open WebUI's model dropdown.
        # Without this step, Open WebUI (which talks to Ollama) won't see
        # GGUF files downloaded directly — Ollama has its own model store.
        if command -v ollama &>/dev/null; then
            # Derive a clean Ollama model tag from the filename.
            # Ollama requires lowercase tags. We separate the quant suffix with ':'
            # e.g. Qwen_Qwen3-8B-Q4_K_M.gguf → qwen_qwen3-8b:q4_k_m
            # sed: case-insensitive match on -Q or -q followed by digit → replace hyphen with colon
            OLLAMA_TAG=$(basename "${M[file]}" .gguf                 | sed -E 's/-([Qq][0-9].*)$/:\1/'                 | tr '[:upper:]' '[:lower:]')

            info "Registering model with Ollama as: $OLLAMA_TAG"
            info "  This lets Open WebUI and 'ollama run' use it."

            MODELFILE_PATH="$TEMP_DIR/Modelfile.$$"
            mkdir -p "$TEMP_DIR"
            cat > "$MODELFILE_PATH" <<MODELFILE
FROM $GGUF_MODELS/${M[file]}
# 999 = Ollama sentinel: "put as many layers on GPU as VRAM allows"
# This is always better than a pre-calculated value because Ollama measures
# actual free VRAM at load time, accounting for driver/CUDA overhead.
PARAMETER num_gpu 999
PARAMETER num_thread $HW_THREADS
# 8192 context — larger than 4096 but still safe for 12 GB VRAM.
# Flash attention + q8_0 KV cache make this affordable even on 6-8 GB cards.
PARAMETER num_ctx 8192
MODELFILE

            if ollama create "$OLLAMA_TAG" -f "$MODELFILE_PATH"; then
                info "✔ Model registered: $OLLAMA_TAG"
                info "  You can now use it in Open WebUI, or run: ollama run $OLLAMA_TAG"
                # Save tag to config so other tools can reference it
                echo "OLLAMA_TAG=\"$OLLAMA_TAG\"" >> "$MODEL_CONFIG"
            else
                warn "ollama create failed — model won't appear in Open WebUI."
                warn "  To register manually:"
                warn "    ollama create $OLLAMA_TAG -f $MODELFILE_PATH"
            fi
            rm -f "$MODELFILE_PATH"
        else
            warn "Ollama not found — skipping model registration."
            warn "  Install Ollama first, then run:"
            warn "    ollama create my-model -f <(echo 'FROM $GGUF_MODELS/${M[file]}')"
        fi
    else
        warn "Download failed. Resume with:"
        warn "  curl -L -C - -o '$GGUF_MODELS/${M[file]}' '${M[url]}'"
    fi
    popd >/dev/null
fi

# =============================================================================
# STEP 12 — HELPER SCRIPTS
# =============================================================================
step "Helper scripts"

# run-gguf: uses hardware-tuned defaults from config
cat > "$BIN_DIR/run-gguf" <<PYEOF
#!/usr/bin/env python3
"""Run a local GGUF model. Defaults loaded from ~/.config/local-llm/selected_model.conf"""
import sys, os, glob, argparse

MODEL_DIR  = os.path.expanduser("~/local-llm-models/gguf")
CONFIG_DIR = os.path.expanduser("~/.config/local-llm")
VENV_SITE  = os.path.expanduser("~/.local/share/llm-venv/lib")

for _sp in glob.glob(os.path.join(VENV_SITE, "python3*/site-packages")):
    if _sp not in sys.path: sys.path.insert(0, _sp)

def load_conf():
    cfg = {}
    p = os.path.join(CONFIG_DIR, "selected_model.conf")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    k, v = line.split('=', 1)
                    cfg[k] = v.strip('"')
    return cfg

def list_models():
    models = glob.glob(os.path.join(MODEL_DIR, "*.gguf"))
    if not models: print("No GGUF models in", MODEL_DIR); return
    print("Available models:")
    for m in sorted(models):
        print(f"  {os.path.basename(m):<55} {os.path.getsize(m)/1024**3:.1f} GB")

def main():
    cfg = load_conf()
    parser = argparse.ArgumentParser(description="Run a GGUF model (auto-tuned to your hardware)")
    parser.add_argument("model",  nargs="?")
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--gpu-layers", type=int,   default=None)
    parser.add_argument("--cpu-layers", type=int,   default=None)
    parser.add_argument("--ctx",        type=int,   default=8192)
    parser.add_argument("--max-tokens", type=int,   default=512)
    parser.add_argument("--threads",    type=int,   default=int(cfg.get("HW_THREADS", 4)))
    parser.add_argument("--batch",      type=int,   default=int(cfg.get("BATCH", 256)))
    args = parser.parse_args()

    if not args.model: list_models(); sys.exit(0)

    model_path = args.model if os.path.isabs(args.model) else os.path.join(MODEL_DIR, args.model)
    if not os.path.exists(model_path):
        print(f"Not found: {model_path}"); list_models(); sys.exit(1)

    prompt     = " ".join(args.prompt) if args.prompt else "Hello! How are you?"
    gpu_layers = args.gpu_layers if args.gpu_layers is not None else int(cfg.get("GPU_LAYERS", 0))
    cpu_layers = args.cpu_layers if args.cpu_layers is not None else int(cfg.get("CPU_LAYERS", 32))

    try:
        from llama_cpp import Llama
        print(f"Loading {os.path.basename(model_path)} | GPU:{gpu_layers} CPU:{cpu_layers} "
              f"threads:{args.threads} batch:{args.batch} ctx:{args.ctx}", flush=True)
        llm = Llama(model_path=model_path, n_gpu_layers=gpu_layers,
                    n_threads=args.threads, n_batch=args.batch,
                    verbose=False, n_ctx=args.ctx)
        out = llm(prompt, max_tokens=args.max_tokens, echo=True, temperature=0.7, top_p=0.95)
        print(out["choices"][0]["text"])
    except ImportError:
        print("ERROR: activate venv first: source ~/.local/share/llm-venv/bin/activate")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__": main()
PYEOF
chmod +x "$BIN_DIR/run-gguf"

cat > "$BIN_DIR/local-models-info" <<'INFOEOF'
#!/usr/bin/env bash
echo "=== Ollama Models ==="; ollama list 2>/dev/null || echo "  (Ollama not running)"
echo ""; echo "=== GGUF Models ==="
shopt -s nullglob; files=(~/local-llm-models/gguf/*.gguf)
if [[ ${#files[@]} -eq 0 ]]; then echo "  (none)"
else for f in "${files[@]}"; do printf "  %-55s %s\n" "$(basename "$f")" "$(du -sh "$f" 2>/dev/null|cut -f1)"; done; fi
echo ""; echo "=== Disk ==="
du -sh ~/local-llm-models 2>/dev/null || echo "  (no models dir)"
if [[ -f ~/.config/local-llm/selected_model.conf ]]; then
    echo ""; echo "=== Active Config ==="
    # shellcheck source=/dev/null
    source ~/.config/local-llm/selected_model.conf
    echo "  Model:      ${MODEL_NAME:-?}  (${MODEL_SIZE:-?})"
    echo "  GPU layers: ${GPU_LAYERS:-?}  CPU layers: ${CPU_LAYERS:-?}"
    echo "  Threads:    ${HW_THREADS:-?}  Batch: ${BATCH:-?}"
    echo "  File:       ${MODEL_FILENAME:-?}"
fi
INFOEOF
chmod +x "$BIN_DIR/local-models-info"
info "Helper scripts written."

# =============================================================================
# STEP 13 — WEB UI
# =============================================================================
step "Web UI"

# Standalone HTML chat UI (zero dependencies — just open in browser)
HTML_UI="$GUI_DIR/llm-chat.html"
python3 - <<'PYEOF_HTML'
import os
html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEURAL TERMINAL</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
:root{
  --bg:#080b0f;--bg2:#0d1117;--bg3:#111820;--bg4:#151e28;
  --border:#1a2535;--border2:#243040;
  --green:#00ff88;--green-dim:#00aa55;--green-dark:#003322;
  --cyan:#00d4ff;--amber:#ffaa00;--red:#ff4455;--purple:#b060ff;
  --text:#b8c8d8;--text-dim:#4a6070;--text-bright:#d8eaf8;
  --glow:0 0 20px rgba(0,255,136,0.25);--glow-sm:0 0 8px rgba(0,255,136,0.15);
  --sidebar:260px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:14px;overflow:hidden}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.06) 2px,rgba(0,0,0,0.06) 4px);pointer-events:none;z-index:9999}

/* ── Layout ────────────────────────────────────────────────────────── */
#shell{display:flex;height:100vh;overflow:hidden}

/* ── LEFT SIDEBAR — sessions ───────────────────────────────────────── */
#sidebar{
  width:var(--sidebar);min-width:var(--sidebar);
  background:var(--bg2);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
  transition:width .2s;
}
#sidebar.collapsed{width:0;min-width:0;border:none}
#sidebar-header{
  padding:14px 12px 10px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  gap:8px;flex-shrink:0;
}
.logo{font-family:'Orbitron',monospace;font-size:14px;font-weight:900;letter-spacing:3px;color:var(--green);text-shadow:var(--glow);white-space:nowrap}
.logo span{color:var(--cyan)}
#new-chat-btn{
  background:var(--green-dark);border:1px solid var(--green-dim);color:var(--green);
  font-family:'JetBrains Mono',monospace;font-size:11px;padding:5px 9px;border-radius:4px;
  cursor:pointer;white-space:nowrap;transition:all .15s;letter-spacing:1px;
}
#new-chat-btn:hover{background:#004422;box-shadow:var(--glow-sm)}
#session-list{flex:1;overflow-y:auto;padding:8px 0;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.session-item{
  padding:9px 12px;cursor:pointer;transition:background .15s;
  border-left:2px solid transparent;display:flex;align-items:center;gap:8px;
}
.session-item:hover{background:var(--bg3)}
.session-item.active{background:var(--bg3);border-left-color:var(--green)}
.session-name{font-size:12px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.session-del{font-size:10px;color:var(--text-dim);padding:2px 4px;border-radius:2px;opacity:0;transition:opacity .15s}
.session-item:hover .session-del{opacity:1}
.session-del:hover{color:var(--red)}

/* ── MAIN AREA ──────────────────────────────────────────────────────── */
#main{display:flex;flex-direction:column;flex:1;min-width:0;overflow:hidden}

/* ── TOP BAR ────────────────────────────────────────────────────────── */
#topbar{
  display:flex;align-items:center;padding:10px 14px;
  border-bottom:1px solid var(--border);gap:10px;flex-shrink:0;flex-wrap:wrap;
  background:var(--bg2);
}
#sidebar-toggle{background:transparent;border:1px solid var(--border);color:var(--text-dim);padding:6px 9px;border-radius:4px;cursor:pointer;font-size:13px;transition:all .15s}
#sidebar-toggle:hover{border-color:var(--green-dim);color:var(--green)}
.status-wrap{display:flex;align-items:center;gap:6px}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse 2s ease-in-out infinite;flex-shrink:0}
.status-dot.offline{background:var(--red);box-shadow:0 0 8px var(--red);animation:none}
.status-dot.connecting{background:var(--amber);box-shadow:0 0 8px var(--amber)}
#status-label{font-size:11px;color:var(--text-dim)}
select{background:var(--bg3);border:1px solid var(--border);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:12px;padding:5px 8px;border-radius:4px;outline:none;cursor:pointer;max-width:220px;transition:border-color .2s}
select:hover,select:focus{border-color:var(--green-dim)}
.stat{font-size:11px;color:var(--text-dim);padding:4px 8px;border:1px solid var(--border);border-radius:4px;white-space:nowrap}
.stat .val{color:var(--cyan)}
.topbar-btns{display:flex;gap:6px;margin-left:auto}
.icon-btn{background:transparent;border:1px solid var(--border);color:var(--text-dim);font-family:'JetBrains Mono',monospace;font-size:11px;padding:5px 9px;border-radius:4px;cursor:pointer;transition:all .15s;white-space:nowrap;letter-spacing:.5px}
.icon-btn:hover{border-color:var(--text-dim);color:var(--text)}
.icon-btn.active{border-color:var(--amber);color:var(--amber)}
#reconnect-btn{border-color:var(--amber);color:var(--amber)}
#reconnect-btn:hover{background:rgba(255,170,0,0.08)}

/* ── SYSTEM PROMPT PANEL ────────────────────────────────────────────── */
#sysprompt-panel{
  border-bottom:1px solid var(--border);background:var(--bg3);
  overflow:hidden;max-height:0;transition:max-height .3s ease;flex-shrink:0;
}
#sysprompt-panel.open{max-height:160px}
#sysprompt-inner{padding:10px 14px}
#sysprompt-label{font-size:11px;color:var(--amber);letter-spacing:2px;text-transform:uppercase;margin-bottom:6px}
#sysprompt{
  width:100%;background:var(--bg2);border:1px solid var(--border);
  color:var(--text);font-family:'JetBrains Mono',monospace;font-size:12px;
  padding:8px 10px;border-radius:4px;resize:none;height:80px;outline:none;
  transition:border-color .2s;line-height:1.5;
}
#sysprompt:focus{border-color:var(--amber)}
#sysprompt::placeholder{color:var(--text-dim)}
#sysprompt-clear{font-size:10px;color:var(--text-dim);padding:2px 6px;border:1px solid var(--border);background:transparent;border-radius:3px;cursor:pointer;margin-top:4px;font-family:'JetBrains Mono',monospace}
#sysprompt-clear:hover{border-color:var(--red);color:var(--red)}

/* ── MESSAGES ───────────────────────────────────────────────────────── */
#messages{overflow-y:auto;padding:20px 16px;display:flex;flex-direction:column;gap:16px;flex:1;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
#messages::-webkit-scrollbar{width:4px}
#messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

.msg{display:grid;grid-template-columns:36px 1fr;gap:12px;animation:fadeIn .2s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
.msg-avatar{width:36px;height:36px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0;margin-top:2px;font-family:'Orbitron',monospace}
.msg.user .msg-avatar{background:var(--green-dark);color:var(--green);border:1px solid var(--green-dim)}
.msg.ai   .msg-avatar{background:#0d1a2a;color:var(--cyan);border:1px solid #1a3a55}
.msg.system-msg .msg-avatar{background:#1a1000;color:var(--amber);border:1px solid #443300;font-size:16px}
.msg-body{min-width:0}
.msg-meta{display:flex;align-items:center;gap:10px;margin-bottom:5px}
.msg-role{font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase}
.msg.user .msg-role{color:var(--green)}.msg.ai .msg-role{color:var(--cyan)}.msg.system-msg .msg-role{color:var(--amber)}
.msg-time{font-size:10px;color:var(--text-dim)}
.msg-actions{display:flex;gap:4px;margin-left:auto}
.msg-btn{font-size:10px;padding:2px 6px;background:transparent;border:1px solid var(--border);color:var(--text-dim);border-radius:3px;cursor:pointer;font-family:'JetBrains Mono',monospace;transition:all .15s}
.msg-btn:hover{border-color:var(--green-dim);color:var(--green)}
.msg-btn.regen{border-color:var(--purple);color:var(--purple)}
.msg-btn.regen:hover{background:rgba(176,96,255,0.1)}
.msg-content{color:var(--text-bright);line-height:1.75;word-break:break-word}
.msg.user .msg-content{background:var(--bg2);border:1px solid var(--border);border-left:3px solid var(--green-dim);padding:10px 14px;border-radius:0 6px 6px 0}
.msg.ai   .msg-content{background:var(--bg2);border:1px solid var(--border);border-left:3px solid #1a4060;padding:10px 14px;border-radius:0 6px 6px 0}
.msg.system-msg .msg-content{background:#0d0a00;border:1px solid #332200;border-left:3px solid var(--amber);padding:8px 12px;border-radius:0 6px 6px 0;font-size:12px;color:var(--amber)}
.msg-content p{margin:0 0 8px}
.msg-content p:last-child{margin-bottom:0}
.msg-content code{background:var(--bg3);border:1px solid var(--border);padding:1px 5px;border-radius:3px;color:var(--amber);font-size:12px}
.msg-content pre{background:#050709 !important;border:1px solid var(--border);border-left:3px solid var(--amber);border-radius:0 6px 6px 6px;margin:10px 0;overflow:hidden}
.msg-content pre .code-header{display:flex;align-items:center;justify-content:space-between;padding:5px 12px;background:#0a0c10;border-bottom:1px solid var(--border)}
.msg-content pre .lang-tag{font-size:10px;color:var(--amber);letter-spacing:1px;text-transform:uppercase}
.msg-content pre .copy-code{font-size:10px;padding:2px 7px;background:transparent;border:1px solid var(--border);color:var(--text-dim);border-radius:3px;cursor:pointer;font-family:'JetBrains Mono',monospace}
.msg-content pre .copy-code:hover{border-color:var(--green-dim);color:var(--green)}
.msg-content pre code{background:none !important;border:none !important;padding:12px 14px !important;display:block;overflow-x:auto;font-size:12px;line-height:1.6}
.msg-content pre code.hljs{background:#050709 !important;padding:12px 14px !important}
.msg-content strong{color:var(--text-bright);font-weight:600}
.msg-content ul,.msg-content ol{margin:6px 0 6px 20px}
.msg-content li{margin-bottom:3px}
.msg-content h1,.msg-content h2,.msg-content h3{color:var(--cyan);margin:12px 0 6px;font-family:'Orbitron',monospace;letter-spacing:1px}
.msg-content h1{font-size:16px}.msg-content h2{font-size:14px}.msg-content h3{font-size:13px}
.msg-content blockquote{border-left:3px solid var(--purple);padding:6px 12px;color:var(--text-dim);background:var(--bg3);margin:8px 0}
.msg-content hr{border:none;border-top:1px solid var(--border);margin:10px 0}
.msg-content table{border-collapse:collapse;width:100%;margin:8px 0;font-size:12px}
.msg-content th,.msg-content td{border:1px solid var(--border);padding:6px 10px;text-align:left}
.msg-content th{background:var(--bg3);color:var(--cyan)}

.cursor::after{content:'▋';color:var(--green);animation:blink .6s step-end infinite}
@keyframes blink{50%{opacity:0}}

#empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:16px;color:var(--text-dim);text-align:center;user-select:none}
.empty-logo{font-family:'Orbitron',monospace;font-size:32px;font-weight:900;color:var(--border);letter-spacing:6px}
.empty-sub{font-size:11px;letter-spacing:2px;text-transform:uppercase}
.suggestion-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;max-width:600px}
.suggestion{background:var(--bg2);border:1px solid var(--border);padding:10px 14px;border-radius:6px;font-size:12px;color:var(--text-dim);cursor:pointer;transition:all .2s;text-align:left}
.suggestion:hover{border-color:var(--green-dim);color:var(--text);background:var(--bg3)}

/* ── INPUT AREA ─────────────────────────────────────────────────────── */
#input-area{border-top:1px solid var(--border);padding:12px 14px 14px;display:flex;flex-direction:column;gap:8px;flex-shrink:0;background:var(--bg2)}
.input-row{display:flex;gap:8px;align-items:flex-end}
#prompt{
  flex:1;background:var(--bg3);border:1px solid var(--border);border-radius:6px;
  color:var(--text-bright);font-family:'JetBrains Mono',monospace;font-size:14px;
  padding:10px 14px;resize:none;outline:none;min-height:44px;max-height:200px;
  line-height:1.5;transition:border-color .2s,box-shadow .2s;
}
#prompt:focus{border-color:var(--green-dim);box-shadow:var(--glow-sm)}
#prompt::placeholder{color:var(--text-dim)}
.btn{background:transparent;border:1px solid var(--border);color:var(--text-dim);font-family:'JetBrains Mono',monospace;font-size:12px;padding:10px 14px;border-radius:6px;cursor:pointer;white-space:nowrap;transition:all .15s;display:flex;align-items:center;gap:6px;letter-spacing:1px;text-transform:uppercase}
#send-btn{background:var(--green-dark);border-color:var(--green-dim);color:var(--green);font-weight:600;min-width:80px;justify-content:center}
#send-btn:hover:not(:disabled){background:#004422;box-shadow:var(--glow-sm)}
#send-btn:disabled{opacity:.35;cursor:not-allowed}
#stop-btn{display:none;border-color:var(--red);color:var(--red)}
#stop-btn:hover{background:rgba(255,68,85,0.08)}
#stop-btn.visible{display:flex}

/* ── PARAMS ROW ─────────────────────────────────────────────────────── */
#params-row{display:flex;align-items:center;gap:16px;flex-wrap:wrap;font-size:11px;color:var(--text-dim)}
.param-group{display:flex;align-items:center;gap:6px}
.param-label{letter-spacing:1px;text-transform:uppercase;white-space:nowrap;font-size:10px}
input[type=range]{padding:0;width:72px;height:3px;accent-color:var(--green);border:none;background:none;cursor:pointer}
.param-val{color:var(--cyan);min-width:28px;text-align:right;font-size:11px}
#max-tokens-input{
  width:58px;background:var(--bg3);border:1px solid var(--border);border-radius:3px;
  color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:11px;
  padding:2px 5px;outline:none;text-align:right;
}
#max-tokens-input:focus{border-color:var(--green-dim)}
.input-footer-right{margin-left:auto;font-size:10px;color:var(--text-dim)}

/* ── PARAMS PANEL TOGGLE ────────────────────────────────────────────── */
#params-toggle{
  background:transparent;border:none;color:var(--text-dim);font-family:'JetBrains Mono',monospace;
  font-size:10px;cursor:pointer;padding:0;letter-spacing:1px;text-transform:uppercase;
  transition:color .15s;
}
#params-toggle:hover{color:var(--text)}
#params-panel{overflow:hidden;max-height:0;transition:max-height .25s ease}
#params-panel.open{max-height:60px}

@media(max-width:650px){
  #sidebar{display:none}
  .stat{display:none}
  .suggestion-grid{grid-template-columns:1fr}
  #params-row{gap:10px}
}
</style>
</head>
<body>
<div id="shell">

  <!-- ── Sidebar ── -->
  <aside id="sidebar">
    <div id="sidebar-header">
      <div class="logo">NEURAL<span>TERM</span></div>
      <button id="new-chat-btn" onclick="newSession()">+ NEW</button>
    </div>
    <div id="session-list"></div>
  </aside>

  <!-- ── Main ── -->
  <div id="main">

    <!-- Topbar -->
    <div id="topbar">
      <button id="sidebar-toggle" onclick="toggleSidebar()" title="Toggle sessions">☰</button>
      <div class="status-wrap">
        <div class="status-dot offline" id="status-dot"></div>
        <span id="status-label">connecting…</span>
      </div>
      <select id="model-select"><option value="">— loading —</option></select>
      <div class="stat">CTX <span class="val" id="token-count">—</span></div>
      <div class="stat">SPEED <span class="val" id="speed-display">—</span></div>
      <div class="topbar-btns">
        <button class="icon-btn" id="sysprompt-btn" onclick="toggleSysprompt()" title="System prompt">⚙ SYSTEM</button>
        <button class="icon-btn" onclick="exportChat()" title="Export conversation as Markdown">↓ EXPORT</button>
        <button class="icon-btn" onclick="clearChat()" title="Clear current chat">✕ CLEAR</button>
        <button class="icon-btn" id="reconnect-btn" onclick="reconnect()" title="Reconnect to Ollama">↻ RECONNECT</button>
      </div>
    </div>

    <!-- System prompt panel -->
    <div id="sysprompt-panel">
      <div id="sysprompt-inner">
        <div id="sysprompt-label">⚙ SYSTEM PROMPT</div>
        <textarea id="sysprompt" placeholder="You are a helpful assistant… (leave blank for model default)"></textarea>
        <button id="sysprompt-clear" onclick="document.getElementById('sysprompt').value=''">✕ CLEAR</button>
      </div>
    </div>

    <!-- Messages -->
    <div id="messages">
      <div id="empty-state">
        <div class="empty-logo">N T</div>
        <div class="empty-sub">Neural Terminal · Local LLM</div>
        <div class="suggestion-grid">
          <div class="suggestion" onclick="useSuggestion(this)">Explain how neural networks learn</div>
          <div class="suggestion" onclick="useSuggestion(this)">Write a Python script to rename files</div>
          <div class="suggestion" onclick="useSuggestion(this)">/think What is the best sorting algorithm?</div>
          <div class="suggestion" onclick="useSuggestion(this)">Summarise the key ideas in this text:</div>
        </div>
      </div>
    </div>

    <!-- Input area -->
    <div id="input-area">
      <div class="input-row">
        <textarea id="prompt" rows="1" placeholder="Send a message… (Enter = send, Shift+Enter = newline)  |  /think for reasoning mode"></textarea>
        <button class="btn" id="stop-btn" onclick="stopGeneration()">■ STOP</button>
        <button class="btn" id="send-btn" onclick="sendMessage()">▶ SEND</button>
      </div>
      <div>
        <button id="params-toggle" onclick="toggleParams()">▸ PARAMETERS</button>
        <div id="params-panel">
          <div id="params-row">
            <div class="param-group">
              <span class="param-label">TEMP</span>
              <input type="range" id="temp" min="0" max="2" step="0.05" value="0.7">
              <span class="param-val" id="temp-val">0.70</span>
            </div>
            <div class="param-group">
              <span class="param-label">TOP-P</span>
              <input type="range" id="topp" min="0" max="1" step="0.01" value="0.95">
              <span class="param-val" id="topp-val">0.95</span>
            </div>
            <div class="param-group">
              <span class="param-label">TOP-K</span>
              <input type="range" id="topk" min="1" max="100" step="1" value="40">
              <span class="param-val" id="topk-val">40</span>
            </div>
            <div class="param-group">
              <span class="param-label">MAX TOK</span>
              <input type="number" id="max-tokens-input" value="2048" min="64" max="32768" step="64">
            </div>
            <div class="input-footer-right">Enter = send · Shift+Enter = newline</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const OLLAMA = 'http://127.0.0.1:11434';

// ── Session management ─────────────────────────────────────────────────────
let sessions = JSON.parse(localStorage.getItem('nt_sessions') || '[]');
let activeId  = localStorage.getItem('nt_active') || null;

if (!sessions.length) { createSession('Chat 1'); }
else if (!activeId || !sessions.find(s=>s.id===activeId)) {
  activeId = sessions[0].id;
}

function createSession(name) {
  const id = 'sess_' + Date.now();
  sessions.push({ id, name: name||'New Chat', history: [] });
  activeId = id;
  saveSessions();
  return id;
}
function saveSessions() {
  localStorage.setItem('nt_sessions', JSON.stringify(sessions));
  localStorage.setItem('nt_active', activeId);
}
function getActive() { return sessions.find(s=>s.id===activeId) || sessions[0]; }
function newSession() {
  const id = createSession('Chat ' + (sessions.length+1));
  activeId = id;
  saveSessions();
  renderSidebar();
  renderMessages();
}
function switchSession(id) {
  activeId = id;
  saveSessions();
  renderSidebar();
  renderMessages();
  stopGeneration();
}
function deleteSession(id, e) {
  e.stopPropagation();
  sessions = sessions.filter(s=>s.id!==id);
  if (!sessions.length) createSession('Chat 1');
  if (activeId===id) activeId = sessions[0].id;
  saveSessions();
  renderSidebar();
  renderMessages();
}
function renderSidebar() {
  const list = document.getElementById('session-list');
  list.innerHTML = sessions.map(s=>`
    <div class="session-item${s.id===activeId?' active':''}" onclick="switchSession('${s.id}')">
      <span class="session-name" title="${s.name}">${s.name}</span>
      <span class="session-del" onclick="deleteSession('${s.id}',event)">✕</span>
    </div>`).join('');
}
function autoRenameSession(text) {
  const sess = getActive();
  if (sess.history.length === 1) {
    sess.name = text.slice(0, 28) + (text.length>28?'…':'');
    saveSessions();
    renderSidebar();
  }
}

// ── Sidebar toggle ─────────────────────────────────────────────────────────
let sidebarOpen = true;
function toggleSidebar() {
  sidebarOpen = !sidebarOpen;
  document.getElementById('sidebar').classList.toggle('collapsed', !sidebarOpen);
}

// ── System prompt toggle ───────────────────────────────────────────────────
let syspromptOpen = false;
function toggleSysprompt() {
  syspromptOpen = !syspromptOpen;
  document.getElementById('sysprompt-panel').classList.toggle('open', syspromptOpen);
  document.getElementById('sysprompt-btn').classList.toggle('active', syspromptOpen);
}

// ── Params panel toggle ────────────────────────────────────────────────────
let paramsOpen = false;
function toggleParams() {
  paramsOpen = !paramsOpen;
  document.getElementById('params-panel').classList.toggle('open', paramsOpen);
  document.getElementById('params-toggle').textContent = (paramsOpen?'▾':'▸') + ' PARAMETERS';
}

// ── Ollama connection ──────────────────────────────────────────────────────
let ollamaOnline = false;
async function loadModels() {
  const dot   = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  dot.className = 'status-dot connecting';
  label.textContent = 'connecting…';
  try {
    const r = await fetch(OLLAMA+'/api/tags', {signal: AbortSignal.timeout(4000)});
    if (!r.ok) throw new Error('HTTP '+r.status);
    const d = await r.json();
    const sel = document.getElementById('model-select');
    sel.innerHTML = '';
    if (!d.models?.length) {
      sel.innerHTML = '<option value="">No models — run: ollama pull qwen3:8b</option>';
    } else {
      d.models.forEach(m=>{
        const o = document.createElement('option');
        o.value = m.name; o.textContent = m.name; sel.appendChild(o);
      });
    }
    dot.className = 'status-dot';
    label.textContent = 'online';
    ollamaOnline = true;
  } catch {
    dot.className = 'status-dot offline';
    label.textContent = 'offline';
    ollamaOnline = false;
    document.getElementById('model-select').innerHTML =
      '<option value="">Ollama offline — run: ollama-start</option>';
  }
}
async function reconnect() {
  document.getElementById('status-label').textContent = 'reconnecting…';
  await loadModels();
}
setInterval(loadModels, 20000);
loadModels();

// ── Slider bindings ────────────────────────────────────────────────────────
['temp','topp','topk'].forEach(id => {
  const el = document.getElementById(id);
  const vl = document.getElementById(id+'-val');
  el.addEventListener('input', () => {
    vl.textContent = id==='topk' ? el.value : parseFloat(el.value).toFixed(2);
  });
});

// ── Prompt box ─────────────────────────────────────────────────────────────
const promptEl = document.getElementById('prompt');
promptEl.addEventListener('input', () => {
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 200) + 'px';
});
promptEl.addEventListener('keydown', e => {
  if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ── Markdown + syntax highlighting renderer ────────────────────────────────
function escHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function renderMarkdown(raw) {
  let t = raw;
  // Code blocks
  t = t.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const trimmed = code.trim();
    const highlighted = lang && hljs.getLanguage(lang)
      ? hljs.highlight(trimmed, {language:lang}).value
      : hljs.highlightAuto(trimmed).value;
    const langLabel = lang || 'code';
    return `<pre><div class="code-header"><span class="lang-tag">${langLabel}</span><button class="copy-code" onclick="copyCode(this)">COPY</button></div><code class="hljs">${highlighted}</code></pre>`;
  });
  // Inline code
  t = t.replace(/`([^`\n]+)`/g, (_, c) => `<code>${escHtml(c)}</code>`);
  // Headers
  t = t.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  t = t.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  t = t.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // Bold / italic
  t = t.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Blockquote
  t = t.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
  // Horizontal rule
  t = t.replace(/^---+$/gm, '<hr>');
  // Tables (simple)
  t = t.replace(/(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)+)/g, tbl => {
    const rows = tbl.trim().split('\n');
    const header = rows[0].split('|').filter(c=>c.trim()).map(c=>`<th>${c.trim()}</th>`).join('');
    const body = rows.slice(2).map(r=>'<tr>'+r.split('|').filter(c=>c.trim()).map(c=>`<td>${c.trim()}</td>`).join('')+'</tr>').join('');
    return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
  });
  // Lists
  t = t.replace(/^(\s*[-*+] .+(\n|$))+/gm, blk => '<ul>'+blk.replace(/^\s*[-*+] (.+)$/gm,'<li>$1</li>')+'</ul>');
  t = t.replace(/^(\s*\d+\. .+(\n|$))+/gm, blk => '<ol>'+blk.replace(/^\s*\d+\. (.+)$/gm,'<li>$1</li>')+'</ol>');
  // Paragraphs (double newline)
  t = t.replace(/\n\n+/g, '</p><p>');
  if (!t.startsWith('<')) t = '<p>' + t + '</p>';
  return t;
}
function copyCode(btn) {
  const code = btn.closest('pre').querySelector('code');
  navigator.clipboard.writeText(code.innerText).then(()=>{
    btn.textContent='COPIED'; setTimeout(()=>btn.textContent='COPY',1500);
  }).catch(()=>{});
}

// ── Message rendering ──────────────────────────────────────────────────────
function now() { return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}); }
function scrollBot() { const m=document.getElementById('messages'); m.scrollTop=m.scrollHeight; }
function setStreaming(on) {
  streaming=on;
  document.getElementById('send-btn').disabled=on;
  document.getElementById('stop-btn').classList.toggle('visible',on);
}

function appendMsg(role, content, stream=false) {
  const empty = document.getElementById('empty-state');
  if (empty) empty.remove();
  const msgs  = document.getElementById('messages');
  const id    = 'msg_'+Date.now()+'_'+Math.random().toString(36).slice(2);
  const isU   = role==='user';
  const div   = document.createElement('div');
  div.className = 'msg '+(isU?'user':'ai');
  div.id = id;
  const actions = isU
    ? `<button class="msg-btn" onclick="copyMsg('${id}')">COPY</button>`
    : `<button class="msg-btn" onclick="copyMsg('${id}')">COPY</button>
       <button class="msg-btn regen" onclick="regenerate('${id}')">↻ REGEN</button>`;
  div.innerHTML = `
    <div class="msg-avatar">${isU?'U':'AI'}</div>
    <div class="msg-body">
      <div class="msg-meta">
        <span class="msg-role">${isU?'USER':'ASSISTANT'}</span>
        <span class="msg-time">${now()}</span>
        <div class="msg-actions">${actions}</div>
      </div>
      <div class="msg-content${stream?' cursor':''}">${isU?escHtml(content):renderMarkdown(content)}</div>
    </div>`;
  msgs.appendChild(div);
  scrollBot();
  return id;
}
function updateMsg(id, content, done=false) {
  const el = document.getElementById(id)?.querySelector('.msg-content');
  if (!el) return;
  el.innerHTML = renderMarkdown(content);
  done ? el.classList.remove('cursor') : el.classList.add('cursor');
  scrollBot();
}
function copyMsg(id) {
  const el = document.getElementById(id)?.querySelector('.msg-content');
  if (el) navigator.clipboard.writeText(el.innerText).catch(()=>{});
}

// ── Regenerate ────────────────────────────────────────────────────────────
function regenerate(msgId) {
  const sess = getActive();
  // Find last user message
  const lastUser = [...sess.history].reverse().find(m=>m.role==='user');
  if (!lastUser) return;
  // Remove last assistant turn from history
  while (sess.history.length && sess.history[sess.history.length-1].role==='assistant')
    sess.history.pop();
  saveSessions();
  // Remove the AI bubble from DOM
  document.getElementById(msgId)?.remove();
  // Re-run the prompt
  runInference(lastUser.content);
}

// ── Send message ──────────────────────────────────────────────────────────
let streaming = false, abortCtrl = null;
function useSuggestion(el) { promptEl.value=el.textContent; promptEl.dispatchEvent(new Event('input')); promptEl.focus(); }

async function sendMessage() {
  const model = document.getElementById('model-select').value;
  const text  = promptEl.value.trim();
  if (!text || !model || streaming) return;
  promptEl.value=''; promptEl.style.height='auto';
  const sess = getActive();
  sess.history.push({role:'user',content:text});
  saveSessions();
  appendMsg('user', text);
  autoRenameSession(text);
  await runInference(text);
}

async function runInference(userText) {
  const model = document.getElementById('model-select').value;
  if (!model) return;
  const sess = getActive();
  const aiId = appendMsg('assistant','',true);
  setStreaming(true);
  abortCtrl = new AbortController();
  const t0 = Date.now();
  let full='', tokenCount=0;

  // Build messages array — prepend system prompt if set
  const sysPrompt = document.getElementById('sysprompt').value.trim();
  const messages = sysPrompt
    ? [{role:'system',content:sysPrompt}, ...sess.history]
    : [...sess.history];

  const opts = {
    temperature: parseFloat(document.getElementById('temp').value),
    top_p:       parseFloat(document.getElementById('topp').value),
    top_k:       parseInt(document.getElementById('topk').value),
    num_predict: parseInt(document.getElementById('max-tokens-input').value)||2048,
  };

  try {
    const res = await fetch(OLLAMA+'/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      signal: abortCtrl.signal,
      body: JSON.stringify({model, messages, stream:true, options:opts})
    });
    if (!res.ok) throw new Error('HTTP '+res.status);
    const reader=res.body.getReader(), dec=new TextDecoder();
    while(true){
      const{done,value}=await reader.read(); if(done) break;
      for(const line of dec.decode(value,{stream:true}).split('\n')){
        if(!line.trim()) continue;
        try{
          const j=JSON.parse(line);
          if(j.message?.content){ full+=j.message.content; updateMsg(aiId,full,false); }
          if(j.eval_count){ tokenCount=j.eval_count; }
          if(j.eval_duration){
            const secs = j.eval_duration/1e9;
            const tps  = secs>0 ? (tokenCount/secs).toFixed(1) : '?';
            document.getElementById('speed-display').textContent = tps+' t/s';
          }
          if(j.done) updateMsg(aiId,full,true);
        }catch{}
      }
    }
    document.getElementById('token-count').textContent = tokenCount+' tok';
  } catch(err){
    if(err.name==='AbortError'){
      updateMsg(aiId, full+'\n\n*[stopped]*', true);
    } else {
      updateMsg(aiId, '**ERROR:** '+err.message+'\n\nIs Ollama running? Try the **↻ RECONNECT** button.', true);
    }
  } finally {
    sess.history.push({role:'assistant',content:full});
    saveSessions();
    setStreaming(false);
    abortCtrl=null;
  }
}

function stopGeneration() { if(abortCtrl){abortCtrl.abort();abortCtrl=null;} setStreaming(false); }

// ── Clear chat ────────────────────────────────────────────────────────────
function clearChat() {
  const sess = getActive();
  sess.history = [];
  saveSessions();
  renderMessages();
  document.getElementById('token-count').textContent = '—';
  document.getElementById('speed-display').textContent = '—';
}

// ── Render messages from session history ──────────────────────────────────
function renderMessages() {
  const msgs = document.getElementById('messages');
  const sess = getActive();
  if (!sess.history.length) {
    msgs.innerHTML = `<div id="empty-state">
      <div class="empty-logo">N T</div>
      <div class="empty-sub">Neural Terminal · Local LLM</div>
      <div class="suggestion-grid">
        <div class="suggestion" onclick="useSuggestion(this)">Explain how neural networks learn</div>
        <div class="suggestion" onclick="useSuggestion(this)">Write a Python script to rename files</div>
        <div class="suggestion" onclick="useSuggestion(this)">/think What is the best sorting algorithm?</div>
        <div class="suggestion" onclick="useSuggestion(this)">Summarise the key ideas in this text:</div>
      </div>
    </div>`;
    return;
  }
  msgs.innerHTML = '';
  sess.history.forEach(m => {
    if (m.role==='system') return; // don't render system messages
    appendMsg(m.role, m.content, false);
  });
}

// ── Export conversation ───────────────────────────────────────────────────
function exportChat() {
  const sess = getActive();
  if (!sess.history.filter(m=>m.role!=='system').length) {
    alert('Nothing to export yet.'); return;
  }
  const model = document.getElementById('model-select').value || 'unknown';
  let md = `# ${sess.name}\n\n**Model:** ${model}  \n**Exported:** ${new Date().toLocaleString()}\n\n---\n\n`;
  sess.history.forEach(m => {
    if (m.role==='system') { md += `> **System:** ${m.content}\n\n`; return; }
    const label = m.role==='user' ? '## 👤 User' : '## 🤖 Assistant';
    md += `${label}\n\n${m.content}\n\n---\n\n`;
  });
  const blob = new Blob([md], {type:'text/markdown'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (sess.name.replace(/[^a-z0-9]/gi,'_').toLowerCase() || 'chat') + '.md';
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Init ──────────────────────────────────────────────────────────────────
renderSidebar();
renderMessages();
promptEl.focus();
</script>
</body>
</html>
"""
path = os.path.expandvars(os.path.expanduser('$HOME/.local/share/llm-webui/llm-chat.html'))
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as f:
    f.write(html)
print(f"HTML UI written to {path}")
PYEOF_HTML

# ── llm-chat launcher ─────────────────────────────────────────────────────────
# WHY HTTP SERVER: browsers block fetch() from file:// to http:// (CORS policy).
# We spin up Python's built-in HTTP server so the page loads from
# http://localhost:8090 — an origin Ollama accepts. No extra deps needed.
cat > "$BIN_DIR/llm-chat" <<'HTMLLAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

GUI_DIR="$HOME/.local/share/llm-webui"
HTML_FILE="$GUI_DIR/llm-chat.html"
HTTP_PORT=8090
BIN_DIR="$HOME/.local/bin"

# ── Check UI file exists ──────────────────────────────────────────────────────
if [[ ! -f "$HTML_FILE" ]]; then
    echo "ERROR: HTML UI not found at $HTML_FILE"
    echo "       Re-run the setup script to regenerate it."
    exit 1
fi

# ── Ensure Ollama is running ──────────────────────────────────────────────────
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
    echo "→ Ollama not running — starting it…"
    if [[ -x "$BIN_DIR/ollama-start" ]]; then
        "$BIN_DIR/ollama-start"
    else
        nohup ollama serve >/dev/null 2>&1 &
    fi
    echo "  Waiting for Ollama to come up…"
    for i in {1..12}; do
        curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
        (( i == 12 )) && echo "  WARNING: Ollama didn't respond in 12s. UI may show 'offline'."
    done
else
    echo "→ Ollama already running."
fi

# ── Kill any stale HTTP server on our port ────────────────────────────────────
OLD_PID=$(lsof -ti tcp:$HTTP_PORT 2>/dev/null || true)
if [[ -n "$OLD_PID" ]]; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 0.5
fi

# ── Start HTTP server in background ──────────────────────────────────────────
echo "→ Starting HTTP server on http://localhost:$HTTP_PORT …"
python3 -m http.server "$HTTP_PORT" \
    --directory "$GUI_DIR" \
    --bind 127.0.0.1 \
    >/dev/null 2>&1 &
HTTP_PID=$!

# Give the server a moment to bind
sleep 0.8

# Verify it's up
if ! kill -0 "$HTTP_PID" 2>/dev/null; then
    echo "ERROR: HTTP server failed to start."
    echo "       Is port $HTTP_PORT already in use? Try: lsof -i :$HTTP_PORT"
    exit 1
fi

URL="http://localhost:$HTTP_PORT/llm-chat.html"
echo "→ Opening $URL"
echo ""
echo "  Press Ctrl+C here to stop the server when done."
echo ""

# ── Open browser ──────────────────────────────────────────────────────────────
if grep -qi microsoft /proc/version 2>/dev/null; then
    # WSL2 — open in Windows default browser
    cmd.exe /c start "" "$URL" 2>/dev/null \
        || powershell.exe -Command "Start-Process '$URL'" 2>/dev/null \
        || echo "  Open manually: $URL"
else
    xdg-open "$URL" 2>/dev/null \
        || sensible-browser "$URL" 2>/dev/null \
        || echo "  Open manually: $URL"
fi

# ── Keep server alive until Ctrl+C ───────────────────────────────────────────
trap "echo ''; echo 'Stopping HTTP server…'; kill $HTTP_PID 2>/dev/null; exit 0" INT TERM
wait "$HTTP_PID"
HTMLLAUNCHER
chmod +x "$BIN_DIR/llm-chat"
info "Web UI: llm-chat  →  serves on http://localhost:8090"

OWUI_VENV="$HOME/.local/share/open-webui-venv"
info "Installing Open WebUI (full ChatGPT-style interface, ~500 MB)…"
echo "  This gives you file uploads, image generation, web search, RAG and more."
echo "  The lightweight Neural Terminal (llm-chat) is still available as a fallback."
echo ""
info "Creating Open WebUI venv…"
[[ ! -d "$OWUI_VENV" ]] && "${PYTHON_BIN:-python3}" -m venv "$OWUI_VENV"
"$OWUI_VENV/bin/pip" install --upgrade pip --quiet || true
info "Installing open-webui (this can take 3-5 minutes)…"
# No --quiet so errors are visible
"$OWUI_VENV/bin/pip" install open-webui \
    || { warn "Open WebUI pip install failed. Check output above."; }

# For WSL2: bind 0.0.0.0 so Windows browser can reach it via localhost
# For native Linux: 127.0.0.1 is fine
if is_wsl2; then
    OWUI_HOST="0.0.0.0"
    OWUI_NOTE="Then open http://localhost:8080 in your Windows browser."
else
    OWUI_HOST="127.0.0.1"
    OWUI_NOTE="Then open http://localhost:8080 in your browser."
fi

cat > "$BIN_DIR/llm-web" <<OWUI_LAUNCHER
#!/usr/bin/env bash
# ── Open WebUI launcher ───────────────────────────────────────────────────────

export DATA_DIR="$GUI_DIR/open-webui-data"
mkdir -p "\$DATA_DIR"

# ── Ollama connection ─────────────────────────────────────────────────────────
export OLLAMA_BASE_URL="http://127.0.0.1:11434"

# ── Disable OpenAI API integration ───────────────────────────────────────────
# Without this, Open WebUI polls api.openai.com on every page load and logs:
#   "Missing bearer authentication in header" (error spam in terminal)
# We're running fully local — no OpenAI key needed or wanted.
export ENABLE_OPENAI_API=false

# ── Ollama OpenAI-compatible shim (fixes "no output" with some model configs) ─
# Open WebUI's chat endpoint can fall through to the /v1/chat/completions path
# even when talking to Ollama. Setting a dummy key + pointing at Ollama's own
# OpenAI-shim endpoint ensures responses flow correctly.
export OPENAI_API_BASE_URL="http://127.0.0.1:11434/v1"
export OPENAI_API_KEY="ollama"   # Ollama accepts any non-empty string here

# ── Audio / misc ──────────────────────────────────────────────────────────────
# Suppress pydub/ffmpeg RuntimeWarning (ffmpeg is installed, just a stale cache)
export PYTHONWARNINGS="ignore::RuntimeWarning"
# Suppress langchain user-agent warning
export USER_AGENT="open-webui/local"
# Restrict CORS to localhost — prevents unnecessary warning on startup
export CORS_ALLOW_ORIGIN="http://localhost:8080"

# ── Start Ollama if not running ───────────────────────────────────────────────
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
    echo "→ Starting Ollama…"
    "$BIN_DIR/ollama-start"
    echo "  Waiting for Ollama API…"
    for i in {1..15}; do
        curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
        (( i == 15 )) && echo "  WARNING: Ollama not responding — UI may show offline."
    done
else
    echo "→ Ollama already running."
fi

# ── Kill anything already on port 8080 ───────────────────────────────────────
OLD_PID=\$(lsof -ti tcp:8080 2>/dev/null || true)
if [[ -n "\$OLD_PID" ]]; then
    echo "→ Port 8080 in use (PID \$OLD_PID) — killing stale process…"
    kill "\$OLD_PID" 2>/dev/null || true
    sleep 1
fi

echo ""
echo "→ Open WebUI starting on port 8080…"
echo "  ${OWUI_NOTE}"
echo "  Press Ctrl+C to stop."
echo ""
"$OWUI_VENV/bin/open-webui" serve --host $OWUI_HOST --port 8080
OWUI_LAUNCHER
chmod +x "$BIN_DIR/llm-web"
info "Open WebUI installed → run: llm-web"
info "  ${OWUI_NOTE}"

# =============================================================================
# STEP — QUALITY-OF-LIFE TOOLS
# =============================================================================
step "Quality-of-life tools"

echo ""
echo -e "  ${CYAN}Tools are grouped so you can skip what you don't need.${NC}"
echo ""

# ── Group 1: Terminal shell (zsh + oh-my-zsh) ─────────────────────────────────
if ask_yes_no "Install Zsh + Oh My Zsh (syntax highlighting, autosuggestions, fzf tab)?"; then
    sudo apt-get install -y zsh zsh-common \
        || warn "zsh install failed."

    if [[ ! -d "$HOME/.oh-my-zsh" ]]; then
        info "Installing Oh My Zsh…"
        sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
            "" --unattended </dev/null \
            || warn "Oh My Zsh install returned non-zero."
    else
        info "Oh My Zsh already installed."
    fi

    ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"
    for _repo in "zsh-users/zsh-syntax-highlighting" "zsh-users/zsh-autosuggestions"; do
        _name="${_repo##*/}"
        _dir="$ZSH_CUSTOM/plugins/$_name"
        if [[ ! -d "$_dir" ]]; then
            retry 2 5 git clone "https://github.com/${_repo}.git" "$_dir"                 || warn "Failed to clone $_name — zsh plugin may not work."
        else
            info "$_name already present."
        fi
    done

    # fzf + fzf-tab for fuzzy completions
    if ask_yes_no "  Also install fzf + fzf-tab (fuzzy file/history/tab search)?"; then
        sudo apt-get install -y fzf 2>/dev/null \
            || { retry 2 5 git clone --depth 1 https://github.com/junegunn/fzf.git "$TEMP_DIR/fzf" \
                && "$TEMP_DIR/fzf/install" --all --no-bash --no-fish --no-update-rc </dev/null; } \
            || warn "fzf install failed."
        _ftdir="$ZSH_CUSTOM/plugins/fzf-tab"
        if [[ ! -d "$_ftdir" ]]; then
            retry 2 5 git clone https://github.com/Aloxaf/fzf-tab "$_ftdir"                 || warn "Failed to clone fzf-tab — tab completions won't work."
        fi
    fi

    # Patch .zshrc with plugins
    if [[ -f "$HOME/.zshrc" ]]; then
        grep -q "zsh-syntax-highlighting" "$HOME/.zshrc" \
            || sed -i 's/^plugins=(\(.*\))/plugins=(\1 zsh-syntax-highlighting zsh-autosuggestions)/' \
               "$HOME/.zshrc" 2>/dev/null || true
        grep -q "fzf-tab" "$HOME/.zshrc" \
            || sed -i 's/^plugins=(\(.*\))/plugins=(\1 fzf-tab)/' "$HOME/.zshrc" 2>/dev/null || true
    fi

    # Note: LLM aliases are added to .zshrc in the ALIASES step below —
    # ALIAS_FILE does not exist yet at this point in the script.

    ZSH_BIN=$(command -v zsh 2>/dev/null || true)
    if [[ -n "$ZSH_BIN" && "$SHELL" != "$ZSH_BIN" ]]; then
        ask_yes_no "  Set zsh as your default shell?" \
            && { chsh -s "$ZSH_BIN" && info "Default shell changed to zsh." \
                 || warn "chsh failed — run manually: chsh -s $ZSH_BIN"; }
    fi
    info "Zsh + Oh My Zsh: done."
fi

# ── Group 2: Terminal multiplexer ─────────────────────────────────────────────
if ask_yes_no "Install tmux (terminal multiplexer — split panes, detach sessions)?"; then
    sudo apt-get install -y tmux \
        || warn "tmux install failed."

    # Write a sensible tmux config if none exists
    if [[ ! -f "$HOME/.tmux.conf" ]]; then
        cat > "$HOME/.tmux.conf" <<'TMUXCFG'
# ── Local LLM tmux config ──────────────────────────────────────────────────
set -g default-terminal "screen-256color"
set -g history-limit 10000
set -g mouse on
set -g base-index 1
set -g pane-base-index 1
set -g status-style 'bg=#1a2535 fg=#00ff88'
set -g status-left '#[bold] 🤖 LLM  '
set -g status-right '#[fg=#00d4ff] %H:%M  #[fg=#00ff88]%d-%b '
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
bind r source-file ~/.tmux.conf \; display "Config reloaded"
# ──────────────────────────────────────────────────────────────────────────
TMUXCFG
        info "tmux config written: ~/.tmux.conf"
    else
        info "~/.tmux.conf already exists — not overwriting."
    fi
fi

# ── Group 3: Modern CLI tools ─────────────────────────────────────────────────
if ask_yes_no "Install modern CLI tools (bat, eza, ripgrep, fd, btop, ncdu, jq, micro)?"; then
    CLI_PKGS=(
        bat            # better cat — syntax highlighted output
        ripgrep        # rg — blazing-fast grep
        fd-find        # fd — simpler find
        btop           # beautiful system/GPU/CPU monitor (replaces htop)
        htop           # classic process viewer
        ncdu           # visual disk usage explorer (great for checking model sizes)
        jq             # JSON processor — useful for API debugging
        tree           # directory tree view
        p7zip-full     # 7z archive support
        unzip zip      # basic archive tools
    )
    sudo apt-get install -y "${CLI_PKGS[@]}" \
        || warn "Some CLI tools failed — continuing."

    # micro — modern terminal text editor (better than nano)
    if ! command -v micro &>/dev/null; then
        info "Installing micro editor…"
        mkdir -p "$BIN_DIR"
        if curl -fsSL https://getmic.ro | bash 2>/dev/null; then
            mv micro "$BIN_DIR/micro" 2>/dev/null \
                || sudo mv micro /usr/local/bin/micro 2>/dev/null || true
            info "micro: OK"
        else
            warn "micro install failed — try: sudo apt-get install micro"
        fi
    else
        info "micro already installed."
    fi

    # eza — modern ls replacement (apt name varies by distro)
    if ! command -v eza &>/dev/null; then
        sudo apt-get install -y eza 2>/dev/null \
            || sudo apt-get install -y exa 2>/dev/null \
            || warn "eza/exa not found in apt — skipping."
    fi

    # nvtop — GPU process monitor — supports NVIDIA, AMD & Intel GPUs
    if (( HAS_GPU )); then
        if ask_yes_no "  Install nvtop (live GPU monitor — watch VRAM usage during inference)?"; then
            sudo apt-get install -y nvtop \
                || warn "nvtop not in apt — try: sudo snap install nvtop"
        fi
    fi

    # Handy aliases for the new tools
    for _rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        [[ -f "$_rc" ]] && ! grep -q "# llm-qol-aliases" "$_rc" && cat >> "$_rc" <<'QOLALIASES'

# ── QoL tool aliases (llm-auto-setup) ─────────────────────────────────────
# shellcheck disable=SC2154
command -v bat       &>/dev/null && alias cat='bat --paging=never'
command -v eza       &>/dev/null && alias ls='eza --icons' && alias ll='eza -la --icons'
command -v btop      &>/dev/null && alias top='btop'
# fd is packaged as 'fdfind' on Debian/Ubuntu — add 'fd' shortcut if missing
command -v fdfind    &>/dev/null && ! command -v fd &>/dev/null && alias fd='fdfind'
# NOTE: we do NOT alias 'find' → 'fd' — they have incompatible syntax and
#       many system scripts depend on POSIX find behaviour.
# llm-qol-aliases
QOLALIASES
    done
    info "Modern CLI tools installed."
fi

# ── Group 4: GUI tools (only if display available) ────────────────────────────
HAVE_DISPLAY=0
[[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]] && HAVE_DISPLAY=1
grep -qi microsoft /proc/version 2>/dev/null && HAVE_DISPLAY=1  # WSL2 with WSLg

if (( HAVE_DISPLAY )); then
    if ask_yes_no "Install GUI tools (Thunar file manager, Mousepad text editor, Meld diff)?"; then
        GUI_PKGS=(
            thunar             # fast GTK file manager — browse model folders easily
            mousepad           # lightweight GTK text editor — edit configs/prompts
            meld               # visual diff/merge tool
            gcolor3            # color picker (handy for theming)
        )
        sudo apt-get install -y "${GUI_PKGS[@]}" \
            || warn "Some GUI packages failed."
        info "GUI tools installed."
        info "  thunar   → file manager"
        info "  mousepad → text editor (mousepad &)"
        info "  meld     → visual diff (meld file1 file2)"
    fi
else
    info "No display detected — skipping GUI tools."
    info "  If running WSL2: install VcXsrv or enable WSLg for GUI support."
fi

# ── Group 5: System info / fun ────────────────────────────────────────────────
if ask_yes_no "Install neofetch (system info banner) and fastfetch?"; then
    sudo apt-get install -y neofetch 2>/dev/null || true
    # fastfetch — faster and more feature-rich than neofetch
    if ! command -v fastfetch &>/dev/null; then
        sudo apt-get install -y fastfetch 2>/dev/null \
            || warn "fastfetch not in apt — try: sudo snap install fastfetch"
    fi
    info "neofetch + fastfetch installed. Run: neofetch  or  fastfetch"
fi

info "Quality-of-life tools step complete."

# =============================================================================
# STEP 13b — AUTONOMOUS COWORKING
# =============================================================================
# Two tools are installed:
#
#  cowork  — Open Interpreter: the LLM can autonomously run code, browse the
#             web, read/write files, and execute shell commands on your machine.
#             Think of it as pair-programming where the AI drives.
#             https://github.com/OpenInterpreter/open-interpreter
#
#  aider   — AI pair programmer tightly integrated with git. Understands your
#             codebase, writes + applies patches, runs tests, commits changes.
#             https://aider.chat
#
# Both are pointed at your local Ollama model via the OpenAI-compatible shim.
# No cloud API keys needed.
# =============================================================================
step "Autonomous coworking"

if ask_yes_no "Install autonomous coworking tools (Open Interpreter + Aider)?"; then

    OI_VENV="$HOME/.local/share/open-interpreter-venv"
    AI_VENV="$HOME/.local/share/aider-venv"

    # ── Open Interpreter ──────────────────────────────────────────────────────
    info "Installing Open Interpreter…"
    [[ ! -d "$OI_VENV" ]] && "${PYTHON_BIN:-python3}" -m venv "$OI_VENV"
    "$OI_VENV/bin/pip" install --upgrade pip --quiet || true
    # setuptools must be installed explicitly — Python 3.12 no longer bundles it
    # in venvs, so open-interpreter's use of pkg_resources raises ModuleNotFoundError.
    "$OI_VENV/bin/pip" install --upgrade setuptools --quiet         || warn "setuptools install failed — cowork may crash on Python 3.12."
    "$OI_VENV/bin/pip" install open-interpreter         || warn "Open Interpreter install failed — check output above."

    # Write cowork launcher — reads OLLAMA_TAG from config at runtime
    cat > "$BIN_DIR/cowork" <<'COWORK_EOF'
#!/usr/bin/env bash
# cowork — autonomous AI coworker via Open Interpreter + local Ollama
# The AI can run code, browse the web, manage files — fully local, no cloud.
set -uo pipefail

OI_VENV="$HOME/.local/share/open-interpreter-venv"
CONFIG="$HOME/.config/local-llm/selected_model.conf"

if [[ ! -x "$OI_VENV/bin/interpreter" ]]; then
    echo "ERROR: Open Interpreter not installed. Re-run llm-auto-setup.sh."
    exit 1
fi

# Load model tag from config
OLLAMA_TAG=""
[[ -f "$CONFIG" ]] && source "$CONFIG" 2>/dev/null || true
OLLAMA_TAG="${OLLAMA_TAG:-qwen_qwen3-14b:q4_k_m}"

# Ensure Ollama is running
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "→ Ollama not running — starting it…"
    command -v ollama-start &>/dev/null && ollama-start || nohup ollama serve >/dev/null 2>&1 &
    for i in {1..15}; do
        curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
fi

echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║          🤖  AUTONOMOUS COWORKER                ║"
echo "  ║  Model  : $OLLAMA_TAG"
echo "  ║  Powered: Open Interpreter + Ollama (local)     ║"
echo "  ║  The AI can run code, browse web, manage files  ║"
echo "  ║  Type 'exit' or Ctrl-D to quit                  ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

# Open Interpreter talks to Ollama via its OpenAI-compatible /v1 shim.
# OPENAI_API_KEY can be any non-empty string — Ollama doesn't validate it.
export OPENAI_API_KEY="ollama"
export OPENAI_API_BASE="http://127.0.0.1:11434/v1"

"$OI_VENV/bin/interpreter"     --model "openai/$OLLAMA_TAG"     --context_window 8192     --max_tokens 4096     --api_base "http://127.0.0.1:11434/v1"     --api_key "ollama"     "$@"
COWORK_EOF
    chmod +x "$BIN_DIR/cowork"
    info "cowork launcher written: $BIN_DIR/cowork"

    # ── Aider ─────────────────────────────────────────────────────────────────
    info "Installing Aider…"
    [[ ! -d "$AI_VENV" ]] && "${PYTHON_BIN:-python3}" -m venv "$AI_VENV"
    "$AI_VENV/bin/pip" install --upgrade pip --quiet || true
    "$AI_VENV/bin/pip" install aider-chat         || warn "Aider install failed — check output above."

    # Write aider launcher
    cat > "$BIN_DIR/aider" <<'AIDER_EOF'
#!/usr/bin/env bash
# aider — AI pair programmer with git integration, powered by local Ollama
set -uo pipefail

AI_VENV="$HOME/.local/share/aider-venv"
CONFIG="$HOME/.config/local-llm/selected_model.conf"

if [[ ! -x "$AI_VENV/bin/aider" ]]; then
    echo "ERROR: Aider not installed. Re-run llm-auto-setup.sh."
    exit 1
fi

OLLAMA_TAG=""
[[ -f "$CONFIG" ]] && source "$CONFIG" 2>/dev/null || true
OLLAMA_TAG="${OLLAMA_TAG:-qwen_qwen3-14b:q4_k_m}"

# Ensure Ollama is running
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "→ Starting Ollama…"
    command -v ollama-start &>/dev/null && ollama-start || nohup ollama serve >/dev/null 2>&1 &
    sleep 3
fi

echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║         🛠  AIDER  AI PAIR PROGRAMMER           ║"
echo "  ║  Model  : $OLLAMA_TAG"
echo "  ║  Powered: Aider + Ollama (fully local)          ║"
echo "  ║  Usage  : aider file.py file2.js  (or no args)  ║"
echo "  ║  Type /help inside aider for commands           ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

export OPENAI_API_KEY="ollama"

"$AI_VENV/bin/aider"     --model "openai/$OLLAMA_TAG"     --openai-api-base "http://127.0.0.1:11434/v1"     --no-auto-commits     "$@"
AIDER_EOF
    chmod +x "$BIN_DIR/aider"
    info "aider launcher written: $BIN_DIR/aider"

    info "Autonomous coworking tools installed."
    info "  cowork  — Open Interpreter (code execution, file ops, web browsing)"
    info "  aider   — AI pair programmer (git-integrated, edit files directly)"
else
    info "Skipping autonomous coworking tools."
fi

# =============================================================================
# STEP 14 — ALIASES
# =============================================================================
step "Shell aliases"

cat > "$ALIAS_FILE" <<'ALIASES_EOF'
# ── Local LLM (auto-setup) ────────────────────────────────────────────────────
alias ollama-list='ollama list'
alias ollama-pull='ollama pull'
alias ollama-run='ollama run'
alias gguf-list='local-models-info'
alias gguf-run='run-gguf'
alias ask='run-gguf'
alias llm-status='local-models-info'
alias chat='llm-chat'
alias webui='llm-web'
alias ai='aider'

run-model() {
    local cfg=~/.config/local-llm/selected_model.conf
    [[ ! -f "$cfg" ]] && echo "No model configured." && return 1
    # Source in a subshell — positional-arg style handles paths with spaces correctly
    local model_file
    model_file=$(bash -c 'source "$1" 2>/dev/null; printf "%s" "$MODEL_FILENAME"' _ "$cfg")
    [[ -z "$model_file" ]] && echo "MODEL_FILENAME not set in config." && return 1
    run-gguf "$model_file" "$@"
}

llm-help() {
    cat <<'HELP'
Local LLM commands:
  ollama-pull <model>    Download an Ollama model
  ollama-run  <model>    Run an Ollama model interactively
  gguf-run <file> [txt]  Run a GGUF model
    --gpu-layers N       GPU layers
    --threads N          CPU threads
    --batch N            Batch size
    --ctx N              Context window
  ask / run-model        Shorthand for default model
  llm-status             Show models + hardware config
  chat / llm-chat        Open Neural Terminal at http://localhost:8090
  webui / llm-web        Start Open WebUI at http://localhost:8080
  cowork                 Open Interpreter — AI that runs code + manages files
  ai / aider             AI pair programmer with git integration
  llm-help               This help
HELP
}
# ─────────────────────────────────────────────────────────────────────────────
ALIASES_EOF

for _rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [[ -f "$_rc" ]] && ! grep -q "source $ALIAS_FILE" "$_rc"; then
        { echo ""; echo "# Local LLM aliases"
          echo "[ -f $ALIAS_FILE ] && source $ALIAS_FILE"; } >> "$_rc"
        info "Aliases added to $_rc"
    fi
done

# =============================================================================
# STEP 15 — FINAL VALIDATION
# =============================================================================
step "Final validation"

PASS=0; WARN_COUNT=0

# ── GPU runtime (CUDA or ROCm) ────────────────────────────────────────────────
if (( HAS_NVIDIA )); then
    CUDA_FOUND=0
    find /usr/local /usr/lib /opt -maxdepth 8 \
        \( -name "libcudart.so.12" -o -name "libcudart.so.12.*" \) 2>/dev/null | grep -q . \
        && CUDA_FOUND=1
    (( !CUDA_FOUND )) && ldconfig -p 2>/dev/null | grep -q 'libcudart\.so\.12' && CUDA_FOUND=1
    if (( CUDA_FOUND )); then
        info "✔ CUDA runtime found."
        PASS=$(( PASS + 1 ))
    else
        warn "✘ libcudart.so.12 not found."
        warn "  Fix: sudo ldconfig && source ~/.bashrc"
        WARN_COUNT=$(( WARN_COUNT + 1 ))
    fi
elif (( HAS_AMD_GPU )); then
    if [[ -d /opt/rocm ]] || command -v rocminfo &>/dev/null; then
        info "✔ ROCm runtime found."
        PASS=$(( PASS + 1 ))
    else
        warn "✘ ROCm not found — AMD GPU won't be used for acceleration."
        warn "  Run the script again to install ROCm, or visit: https://rocm.docs.amd.com"
        WARN_COUNT=$(( WARN_COUNT + 1 ))
    fi
else
    info "✔ CPU-only mode — no GPU runtime needed."
    PASS=$(( PASS + 1 ))
fi

# ── llama-cpp-python ──────────────────────────────────────────────────────────
if "$VENV_DIR/bin/python3" -c "import llama_cpp" 2>/dev/null; then
    info "✔ llama-cpp-python OK."
    PASS=$(( PASS + 1 ))
else
    warn "✘ llama-cpp-python import failed."
    if (( HAS_NVIDIA )); then
        warn "  This may be a CUDA library path issue. Try:"
        warn "    sudo ldconfig && source ~/.bashrc"
    elif (( HAS_AMD_GPU )); then
        warn "  This may be a ROCm/HIP library path issue. Try:"
        warn "    source ~/.bashrc  (ROCm libs should be in LD_LIBRARY_PATH)"
        warn "    hipconfig --version  (checks ROCm install)"
    fi
    warn "    $VENV_DIR/bin/python3 -c 'import llama_cpp'"
    WARN_COUNT=$(( WARN_COUNT + 1 ))
fi

# ── Ollama ────────────────────────────────────────────────────────────────────
if is_wsl2; then
    if pgrep -f "ollama serve" >/dev/null 2>&1; then
        info "✔ Ollama running."
        PASS=$(( PASS + 1 ))
    else
        warn "✘ Ollama not running — start with: ollama-start"
        WARN_COUNT=$(( WARN_COUNT + 1 ))
    fi
else
    if systemctl is-active --quiet ollama 2>/dev/null; then
        info "✔ Ollama service active."
        PASS=$(( PASS + 1 ))
    else
        warn "✘ Ollama service not active."
        warn "  Fix: sudo systemctl start ollama"
        warn "  Logs: sudo journalctl -u ollama -n 30"
        WARN_COUNT=$(( WARN_COUNT + 1 ))
    fi
fi

# ── Ollama API reachable ──────────────────────────────────────────────────────
sleep 1
if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    info "✔ Ollama API responding on port 11434."
    PASS=$(( PASS + 1 ))
else
    warn "✘ Ollama API not reachable on port 11434."
    warn "  The HTML chat UI and Open WebUI need this to work."
    warn "  Fix: ollama-start  (then wait 5 sec and try again)"
    WARN_COUNT=$(( WARN_COUNT + 1 ))
fi

# ── Helper scripts ────────────────────────────────────────────────────────────
if [[ -x "$BIN_DIR/run-gguf" ]]; then
    info "✔ run-gguf OK."
    PASS=$(( PASS + 1 ))
else
    warn "✘ run-gguf missing from $BIN_DIR."
    WARN_COUNT=$(( WARN_COUNT + 1 ))
fi

if [[ -x "$BIN_DIR/llm-chat" ]]; then
    info "✔ llm-chat launcher OK."
    PASS=$(( PASS + 1 ))
else
    warn "✘ llm-chat launcher missing."
    WARN_COUNT=$(( WARN_COUNT + 1 ))
fi

if [[ -f "$GUI_DIR/llm-chat.html" ]]; then
    info "✔ HTML UI written."
    PASS=$(( PASS + 1 ))
else
    warn "✘ HTML UI missing from $GUI_DIR."
    WARN_COUNT=$(( WARN_COUNT + 1 ))
fi

if [[ -f "$ALIAS_FILE" ]]; then
    info "✔ Aliases file OK."
    PASS=$(( PASS + 1 ))
else
    warn "✘ Aliases file missing."
    WARN_COUNT=$(( WARN_COUNT + 1 ))
fi

# =============================================================================
# SUMMARY
# =============================================================================

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
if (( WARN_COUNT == 0 )); then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   🚀  Local LLM Auto-Setup — Installation Complete!         ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║   ⚠   Setup complete — ${WARN_COUNT} warning(s) (see below)            ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════╝${NC}"
fi
echo ""
echo -e "    Checks passed : ${GREEN}$PASS${NC}   │   Warnings: ${YELLOW}$WARN_COUNT${NC}   │   Log: $LOG_FILE"

# ── Hardware + model info ─────────────────────────────────────────────────────
echo ""
echo -e "  ${CYAN}┌─────────────────────────  YOUR SETUP  ──────────────────────────┐${NC}"
printf "  ${CYAN}│${NC}  %-16s  %-43s${CYAN}│${NC}\n" "CPU"   "$CPU_MODEL"
printf "  ${CYAN}│${NC}  %-16s  %-43s${CYAN}│${NC}\n" "RAM"   "${TOTAL_RAM_GB} GB"
if (( HAS_NVIDIA )); then
    printf "  ${CYAN}│${NC}  %-16s  %-43s${CYAN}│${NC}\n" "GPU" "$GPU_NAME  (${GPU_VRAM_GB} GB VRAM) [CUDA]"
elif (( HAS_AMD_GPU )); then
    printf "  ${CYAN}│${NC}  %-16s  %-43s${CYAN}│${NC}\n" "GPU" "$GPU_NAME  (${GPU_VRAM_GB} GB VRAM) [ROCm]"
else
    printf "  ${CYAN}│${NC}  %-16s  %-43s${CYAN}│${NC}\n" "GPU" "None (CPU-only)"
fi
if [[ -f "$MODEL_CONFIG" ]]; then
    # shellcheck source=/dev/null
    source "$MODEL_CONFIG" 2>/dev/null || true
    echo -e "  ${CYAN}├────────────────────────────────────────────────────────────────┤${NC}"
    printf "  ${CYAN}│${NC}  %-16s  %-43s${CYAN}│${NC}\n" "Model"      "${MODEL_NAME:-?}"
    printf "  ${CYAN}│${NC}  %-16s  %-43s${CYAN}│${NC}\n" "Caps"       "${MODEL_CAPS:-none}"
    printf "  ${CYAN}│${NC}  %-16s  %-43s${CYAN}│${NC}\n" "GPU layers" "${GPU_LAYERS:-?} / ${MODEL_LAYERS:-?}   threads: ${HW_THREADS:-?}   batch: ${BATCH:-?}"
fi
echo -e "  ${CYAN}└────────────────────────────────────────────────────────────────┘${NC}"

# ── Command reference ─────────────────────────────────────────────────────────
echo ""
echo -e "  ${CYAN}┌──────────────────────────  COMMANDS  ───────────────────────────┐${NC}"
echo -e "  ${CYAN}│${NC}                                                                ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}  ${MAGENTA}── Chat interfaces ─────────────────────────────────────────${NC}  ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}chat${NC}          Open Neural Terminal UI → http://localhost:8090 ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}webui${NC}         Open WebUI (full UI)    → http://localhost:8080 ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}                                                                ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}  ${MAGENTA}── Run models ──────────────────────────────────────────────${NC}  ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}run-model${NC}     Run your default model from the command line    ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}ask${NC}           Alias for run-model                            ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}ollama-run${NC}    Run any Ollama model  (ollama-run <tag>)         ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}ollama-pull${NC}   Download a new Ollama model                     ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}                                                                ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}  ${MAGENTA}── Autonomous coworking ────────────────────────────────────${NC}  ${CYAN}│${NC}"
if [[ -x "$BIN_DIR/cowork" ]]; then
echo -e "  ${CYAN}│${NC}   ${YELLOW}cowork${NC}        AI that writes & runs code, edits files        ${CYAN}│${NC}"
else
echo -e "  ${CYAN}│${NC}   ${YELLOW}cowork${NC}        ${YELLOW}(not installed — re-run setup to add)${NC}         ${CYAN}│${NC}"
fi
if [[ -x "$BIN_DIR/aider" ]]; then
echo -e "  ${CYAN}│${NC}   ${YELLOW}ai / aider${NC}    AI pair programmer with git integration        ${CYAN}│${NC}"
else
echo -e "  ${CYAN}│${NC}   ${YELLOW}ai / aider${NC}    ${YELLOW}(not installed — re-run setup to add)${NC}         ${CYAN}│${NC}"
fi
echo -e "  ${CYAN}│${NC}                                                                ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}  ${MAGENTA}── Ollama management ───────────────────────────────────────${NC}  ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}ollama-start${NC}  Start the Ollama backend                        ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}ollama-list${NC}   List all downloaded Ollama models               ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}llm-status${NC}    Show models, disk usage, and config             ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}gguf-run${NC}      Run a raw GGUF file directly via llama-cpp      ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}gguf-list${NC}     List all downloaded GGUF files                  ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}                                                                ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}  ${MAGENTA}── Help ─────────────────────────────────────────────────────${NC}  ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}   ${YELLOW}llm-help${NC}      Show full command reference                     ${CYAN}│${NC}"
echo -e "  ${CYAN}│${NC}                                                                ${CYAN}│${NC}"
echo -e "  ${CYAN}└────────────────────────────────────────────────────────────────┘${NC}"

# ── First steps ───────────────────────────────────────────────────────────────
echo ""
echo -e "  ${GREEN}  First steps:${NC}"
echo -e "    1.  Open a new terminal  ${CYAN}(aliases load automatically)${NC}"
echo -e "    2.  ${YELLOW}chat${NC}            → browser chat at http://localhost:8090"
echo -e "    3.  ${YELLOW}webui${NC}           → full Open WebUI at http://localhost:8080"
echo -e "    4.  ${YELLOW}run-model${NC}       → quick CLI test of your model"
is_wsl2 && echo -e "    ⚠  After reboot, run ${YELLOW}ollama-start${NC} before opening any UI"
echo ""

# ── Troubleshooting ───────────────────────────────────────────────────────────
if (( WARN_COUNT > 0 )); then
    echo -e "  ${YELLOW}┌──────────────────────  TROUBLESHOOTING  ────────────────────┐${NC}"
    (( HAS_NVIDIA )) && \
    echo -e "  ${YELLOW}│${NC}  CUDA not found  →  sudo ldconfig && source ~/.bashrc       ${YELLOW}│${NC}"
    (( HAS_AMD_GPU )) && \
    echo -e "  ${YELLOW}│${NC}  ROCm not found  →  source ~/.bashrc  (then: hipconfig -v)  ${YELLOW}│${NC}"
    echo -e "  ${YELLOW}│${NC}  Ollama offline  →  ollama-start                            ${YELLOW}│${NC}"
    echo -e "  ${YELLOW}│${NC}  UI won't load   →  ollama-start, wait 5 s, reopen browser  ${YELLOW}│${NC}"
    echo -e "  ${YELLOW}│${NC}  llama-cpp err   →  source ~/.bashrc && run-model hello      ${YELLOW}│${NC}"
    echo -e "  ${YELLOW}│${NC}  cowork crash    →  re-run setup (setuptools will reinstall) ${YELLOW}│${NC}"
    echo -e "  ${YELLOW}└──────────────────────────────────────────────────────────────┘${NC}"
    echo ""
fi

echo -e "  🚀  Enjoy your local LLM!"
echo ""

# ── Clean up sudo keepalive ───────────────────────────────────────────────────
kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
# Reset trap to default so script exits cleanly
trap - EXIT INT TERM
