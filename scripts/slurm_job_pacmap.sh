#!/bin/bash
#SBATCH --job-name=pacmap-job
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass24-cpu

module load miniconda3

source activate /net/pr2/projects/plgrid/plgglscclass/.conda/envs/tf2-gpu
conda activate visualisations-venv

cd $SLURM_SUBMIT_DIR

python3 visualisations/visualisations_script_pacmap.py input.csv output.csv {{n_components}} {{n_neighbors}} 