#!/bin/bash
#SBATCH --job-name=SearchIQL
#SBATCH --output=/home/u021427/logs/RESCO/search_iql_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=45
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
ALGO=iql_ns
CPUS=45
SEEDS=3
python search.py run --config=search.config.iql_ns.yaml --seeds "${SEEDS}" locally --cpus "${CPUS}"
