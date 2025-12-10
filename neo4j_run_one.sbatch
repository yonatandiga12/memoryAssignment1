#!/bin/bash
#SBATCH --partition=main              # partition to run on
#SBATCH --time=0-30:00:00             # max runtime (in hours)
#SBATCH --job-name=dual_agents_job    # job name
#SBATCH --output=job-%j.out           # output log file, %j = job ID
#SBATCH --gpus=1                      # request 1 GPU (if Ollama uses GPU)
#SBATCH --mem=16G                     # request 16 GB RAM
#SBATCH --cpus-per-task=4             # request 4 CPUs

#SBATCH --output=/home/digayona/langgraph_examples/logs/slurm-%j.out

# 1) Load conda and activate your environment
module load anaconda
source activate my_env   # replace with your env name

# 2) Make sure Ollama daemon is running on this node
#    We'll start it in the background for the job.
ollama serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!
sleep 5   # give the server a moment to start

# 3) Ensure the model you want is pulled (done once; cached afterwards)
#ollama pull  llama3.2:3b

# 4) Run your Python code
python memoryAss1/extract_only_one.py

# 5) Stop the Ollama daemon
kill $OLLAMA_PID || true
wait $OLLAMA_PID || true