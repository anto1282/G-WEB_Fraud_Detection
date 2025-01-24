#!/bin/bash

# Configurable Variables
SWEEP_ID="your_default_sweep_id"  # Default sweep ID; can be overridden with an argument
AGENTS=4                          # Number of agents to launch
QUEUE="gpuv100"                   # Queue to submit jobs
JOB_NAME="SweepAgent"             # Base name for jobs
TIME_LIMIT="2:00"                 # Wall time limit (hh:mm)
MEMORY="10GB"                     # Memory per job
EMAIL="your_email@domain.com"     # Email for notifications
VENV_PATH="/dtu/blackhole/0e/154958/miniconda3/bin/activate MLOPS"  # Path to virtual env
ENTITY="s203557-danmarks-tekniske-universitet-dtu"
WORKDIR="~/mlops/G-WEB_Fraud_Detection/src/gweb/"

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --sweep_id) SWEEP_ID="$2"; shift ;;
        --agents) AGENTS="$2"; shift ;;
        --queue) QUEUE="$2"; shift ;;
        --time) TIME_LIMIT="$2"; shift ;;
        --memory) MEMORY="$2"; shift ;;
        --email) EMAIL="$2"; shift ;;
        --entity) ENTITY="$2"; shift ;;
        --workdir) WORKDIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Check if sweep ID is provided
if [[ -z "$SWEEP_ID" ]]; then
    echo "Error: Sweep ID is required. Use --sweep_id to specify it."
    exit 1
fi

echo "Launching $AGENTS agents for sweep ID: $SWEEP_ID"

# Generate and submit BSUB scripts for each agent
for ((i = 1; i <= AGENTS; i++)); do
    JOB_SCRIPT="submit_agent_$i.sh"
    echo "#!/bin/bash
#BSUB -q $QUEUE
#BSUB -J $JOB_NAME-$i
#BSUB -n 4
#BSUB -gpu \"num=1:mode=exclusive_process\"
#BSUB -W $TIME_LIMIT
#BSUB -R \"rusage[mem=$MEMORY]\"
#BSUB -u $EMAIL
#BSUB -B
#BSUB -env "LSB_JOB_REPORT_MAIL=N"
#BSUB -N
#BSUB -o agent_$i_%J.out
#BSUB -e agent_$i_%J.err

# Load necessary modules
module load python3/3.11.4
module load cuda/11.3

# Activate virtual environment
source $VENV_PATH

# Set working directory, delete old tmp files, create new writable folder 
cd $WORKDIR
rm -rf tmp_agent_$i
mkdir tmp_agent_$i
export WANDB_DIR=tmp_agent_$i

# Run WandB agent
wandb agent $ENTITY/G-Web-Fraud-Detection/$SWEEP_ID
" > $JOB_SCRIPT

    # Submit the job
    bsub < $JOB_SCRIPT

    # Optionally remove the script after submission
    rm $JOB_SCRIPT
done

echo "All $AGENTS agents have been submitted to the queue."
