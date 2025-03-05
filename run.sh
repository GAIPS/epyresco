#!/bin/bash

# for e in "${envs[@]}"
# do
for i in {0..4}
do
  python src/main.py --config=iql_ns --env-config=resco with env_args.key=resco_benchmark:cologne8-iql_ns-v1  env_args.tr=$i --force
  echo "Running with IQL_NS and cologne8 for seed=$i"
  sleep 2s
done
for i in {0..4}
do
  python src/main.py --config=iql_ns --env-config=resco with env_args.key=resco_benchmark:ingolstadt7-iql_ns-v1  env_args.tr=$i --force
  echo "Running with IQL_NS and ingolstadt7 for seed=$i"
  sleep 2s
done
# done
