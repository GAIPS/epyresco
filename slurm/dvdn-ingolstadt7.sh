#!/bin/bash
#SBATCH --job-name=IngolstadtCorridorDVDN
#SBATCH --output=/home/u021427/logs/RESCO/ingolstadt_corridor_dvdn_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=12:00:00

## Activate pyenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate RESCO


## Export
export LIBSUMO_AS_TRACI=1

## Init job
ALGO=dvdn_ns
MAP=ingolstadt7
TASK=resco_benchmark:"${MAP}"-"${ALGO}"-v1
STEPS=30
for i in {0..4}
do
   python src/main.py --config="${ALGO}" --env-config=resco with env_args.key="${TASK}" env_args.tr="$i" n_consensus_steps="${STEPS}" --force
   echo "Running with ${ALGO} and ${TASK} trace $i"
   sleep 2s
done
