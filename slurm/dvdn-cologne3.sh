#!/bin/bash
#SBATCH --job-name=CologneCorridorDVDN
#SBATCH --output=/home/u021427/logs/RESCO/cologne_corridor_dvdn_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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
MAP=cologne3
TASK=resco_benchmark:"${MAP}"-"${ALGO}"-v1
for i in {5..9}
do
   python src/main.py --config="${ALGO}" --env-config=resco with env_args.key="${TASK}" env_args.tr="$i" max_distance=4 --force
   echo "Running with ${ALGO} and ${TASK} trace $i"
   sleep 2s
done
