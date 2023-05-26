#!/bin/bash

#SBATCH --job-name=run_yamnet_plot
#SBATCH --output=run_yamnet_plot%j.out
#SBATCH --error=run_yamnet_plot%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load anaconda3
# Create a conda environment (if it doesn't exist)
if [ ! -d "/users/mbourahl/anaconda3/envs/base_env" ]; then
  conda create --name base_env python=3.8 -y
fi

# Activate the conda environment
conda init bash
source ~/.bashrc
conda activate base_env

# Install the required packages (if not already installed)
pip install -r requirements.txt

# Run the Python script
python yamnet_analysis.py --data_dir /scratch/mbourahl/socialbit/ear_dse_yamnet_tv/processed_audio_data.csv --plot_dir /users/mbourahl/projects/yamnet/plot_dir