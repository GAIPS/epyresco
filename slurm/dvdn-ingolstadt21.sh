#!/bin/bash
#SBATCH --job-name=IngolstadtRegionDVDN
#SBATCH --output=/home/u021427/logs/RESCO/ingolstadt_region_dvdn_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150GB
#SBATCH --time=48:00:00

## Activate pyenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate RESCO


## Export
unset LIBSUMO_AS_TRACI=1
export LIBTRACI_AS_TRACI=1

## Change to src
cd src

## Init job
ALGO=dvdn_ns
MAP=ingolstadt21
CPUS=1
SEEDS=1
TASK=resco_benchmark:"${MAP}"-"${ALGO}"-v1
python train.py run --config=train.config.dvdn_ns.yaml --seeds "${SEEDS}" locally --cpus "${CPUS}"
