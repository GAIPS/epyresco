#!/bin/bash
#SBATCH --job-name=Arterial4x4IQL
#SBATCH --output=/home/u021427/logs/RESCO/arterial4x4_iql_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

## Activate pyenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate RESCO


## Export
export LIBSUMO_AS_TRACI=1

## Init job
ALGO=iql_ns
MAP=arterial4x4
TASK=resco_benchmark:"${MAP}"-"${ALGO}"-v1
for i in {0..4}
do
   python src/main.py --config="${ALGO}" --env-config=resco with env_args.key="${TASK}" env_args.tr="$i" --force
   echo "Running with ${ALGO} and ${TASK} trace $i"
   sleep 2s
done
