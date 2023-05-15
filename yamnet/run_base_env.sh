#!/bin/bash

#SBATCH --job-name=setup_env
#SBATCH --output=setup_env%j.out
#SBATCH --error=setup_env%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1:00:00

# Specify the name of the environment and the Python version
env_name="base_env"
python_version="3.8"

# Load Anaconda module (uncomment the following line if your system uses module software management)
# module load anaconda3

# Check if the conda environment exists
env_exists=$(conda info --envs | awk '{print $1}' | grep -Fxq "$env_name" && echo true || echo false)

# Create the conda environment if it does not exist
if [ "$env_exists" = false ]; then
    echo "Creating conda environment..."
    conda create --name $env_name python=$python_version -y
else
    echo "Conda environment $env_name already exists."
fi

# Activate the conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $env_name

# Install the required packages
echo "Installing required packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt file found."
fi

echo "Environment setup complete."
