#!/bin/bash
#SBATCH --job-name=CologneCorridorSearchVDN
#SBATCH --output=/home/u021427/logs/RESCO/search_vdn_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=80GB
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
ALGO=vdn_ns
CPUS=20
SEEDS=3
python search.py run --config=search.config.vdn_ns.yaml --seeds "${SEEDS}" locally --cpus "${CPUS}"
