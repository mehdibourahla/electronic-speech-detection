#!/bin/bash

#SBATCH --job-name=run_yamnet_features
#SBATCH --output=run_yamnet_features%j.out
#SBATCH --error=run_yamnet_features%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load anaconda3

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate base_env

# Run the Python script
python yamnet_features.py --audio_dir /projects/mcshin_research/ear-data/DSE/FixAllWavFiles --output_dir /scratch/mbourahl/socialbit/ear_dse_yamnet_embeddings
