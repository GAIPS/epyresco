#!/bin/bash
#SBATCH --job-name=CologneCorridorSearchDVDN
#SBATCH --output=/home/u021427/logs/RESCO/search_dvdn_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35
#SBATCH --mem=350GB
#SBATCH --time=96:00:00

## Activate pyenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate RESCO


## Export
unset LIBTRACI_AS_TRACI
export LIBSUMO_AS_TRACI=1

## Change to src
cd src

## Init job
ALGO=dvdn_ns
CPUS=35
SEEDS=3
python search.py run --config=search.config.dvdn_ns.yaml --seeds "${SEEDS}" locally --cpus "${CPUS}"
