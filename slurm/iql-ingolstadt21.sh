#!/bin/bash
#SBATCH --job-name=IngolstadtRegionIQL
#SBATCH --output=/home/u021427/logs/RESCO/ingolstadt_region_iql_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150GB
#SBATCH --time=36:00:00

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
ALGO=iql_ns
MAP=ingolstadt21
CPUS=5
SEEDS=1
TASK=resco_benchmark:"${MAP}"-"${ALGO}"-v1

python train.py run --config=train.config.iql_ns.yaml --seeds "${SEEDS}" locally --cpus "${CPUS}"
