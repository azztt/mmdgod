#!/bin/bash
# Script to wait for GPUs to become free and then run training
# Usage: ./wait_and_run.sh
# Or with custom command: ./wait_and_run.sh "0,1,2,3" your_command...

set -e

# Configuration
GPU_IDS="${1:-0,1,2,3}"           # Which GPUs to monitor (comma-separated)
UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-5}"    # GPU utilization threshold (%)
MEM_THRESHOLD="${GPU_MEM_THRESHOLD:-1000}"   # GPU memory threshold (MiB)
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"       # How often to check (seconds)
STABLE_CHECKS="${STABLE_CHECKS:-3}"          # How many consecutive checks must pass

# Default training command if none provided
DEFAULT_CMD="python tools/train_rgbd.py --config-file configs/rgbd/Base_RGBD_BasicAug.yaml --num-gpus 4 --wandb-project cube-dgod --wandb-name cube_rgbd_base_aug"

# Check if command was provided as arguments
if [ "$#" -gt 1 ]; then
    shift  # Remove first argument (GPU_IDS)
    COMMAND="$@"
elif [ "$#" -eq 1 ] && [[ "$1" != *","* ]]; then
    # Single argument that's not GPU IDs - treat as part of default
    COMMAND="$DEFAULT_CMD"
    GPU_IDS="0,1,2,3"
else
    if [ "$#" -eq 1 ]; then
        shift  # Remove GPU_IDS argument
    fi
    COMMAND="$DEFAULT_CMD"
fi

# Change to working directory
cd /mnt/data/users/anweshan/omni3d

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cube

echo "=========================================="
echo "GPU Wait-and-Run Script"
echo "=========================================="
echo "Monitoring GPUs: $GPU_IDS"
echo "Utilization threshold: <${UTIL_THRESHOLD}%"
echo "Memory threshold: <${MEM_THRESHOLD} MiB"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Required stable checks: ${STABLE_CHECKS}"
echo "Command to run: $COMMAND"
echo "=========================================="
echo ""

# Function to check if GPUs are free
check_gpus_free() {
    local gpus="$1"
    local all_free=true
    
    # Query GPU utilization and memory for specified GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$gpus"
    
    for gpu_id in "${GPU_ARRAY[@]}"; do
        # Get utilization and memory usage
        local info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null)
        
        if [ -z "$info" ]; then
            echo "  GPU $gpu_id: Error querying"
            all_free=false
            continue
        fi
        
        local util=$(echo "$info" | cut -d',' -f1 | tr -d ' ')
        local mem=$(echo "$info" | cut -d',' -f2 | tr -d ' ')
        
        if [ "$util" -ge "$UTIL_THRESHOLD" ] || [ "$mem" -ge "$MEM_THRESHOLD" ]; then
            echo "  GPU $gpu_id: BUSY (util=${util}%, mem=${mem}MiB)"
            all_free=false
        else
            echo "  GPU $gpu_id: FREE (util=${util}%, mem=${mem}MiB)"
        fi
    done
    
    $all_free
}

# Function to show current GPU status
show_gpu_status() {
    echo ""
    echo "Current GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | while read line; do
        echo "  $line"
    done
    echo ""
}

# Main loop
stable_count=0
check_num=0

echo "Starting GPU monitoring at $(date)"
echo ""

while true; do
    check_num=$((check_num + 1))
    echo "[Check #$check_num @ $(date '+%H:%M:%S')] Checking GPU availability..."
    
    if check_gpus_free "$GPU_IDS"; then
        stable_count=$((stable_count + 1))
        echo "  -> All GPUs free! (${stable_count}/${STABLE_CHECKS} stable checks)"
        
        if [ "$stable_count" -ge "$STABLE_CHECKS" ]; then
            echo ""
            echo "=========================================="
            echo "GPUs are free! Starting training..."
            echo "Time: $(date)"
            echo "=========================================="
            echo ""
            
            # Export CUDA_VISIBLE_DEVICES if needed
            export CUDA_VISIBLE_DEVICES="$GPU_IDS"
            
            # Run the command
            exec $COMMAND
        fi
    else
        if [ "$stable_count" -gt 0 ]; then
            echo "  -> GPUs became busy again, resetting counter"
        fi
        stable_count=0
    fi
    
    echo "  Waiting ${CHECK_INTERVAL}s before next check..."
    echo ""
    sleep "$CHECK_INTERVAL"
done
