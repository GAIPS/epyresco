#!/bin/bash
#SBATCH --job-name=SearchIQL
#SBATCH --output=/home/u021427/logs/RESCO/search_iql_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=240GB
#SBATCH --time=72:00:00

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
ALGO=iql_ns
CPUS=25
SEEDS=3
python search.py run --config=search.config.iql_ns.yaml --seeds "${SEEDS}" locally --cpus "${CPUS}"
