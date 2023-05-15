#!/bin/bash

#SBATCH --job-name=run_yamnet_tv
#SBATCH --output=run_yamnet_tv%j.out
#SBATCH --error=run_yamnet_tv%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
python yamnet_pretrained.py --audio_dir /projects/mcshin_research/ear-data/DSE/FixAllWavFiles --gt_dir /projects/mcshin_research/ear-data/Data_Materials/DSE_Raw_Data/Fixed_DSE_EAR_RawData.csv --output_dir /scratch/mbourahl/socialbit/ear_dse_yamnet_tv
