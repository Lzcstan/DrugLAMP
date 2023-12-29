#!/bin/bash

devices=$1

# Define the Python script to run
SCRIPT="main.py"

# Define the arguments for each run
ARGS=(
  "--model DrugLAMPwoSSL --data human --split random --seed 40"
  "--model DrugLAMPwoSSL --data human --split random --seed 41"
  "--model DrugLAMPwoSSL --data human --split random --seed 42"
  "--model DrugLAMPwoSSL --data human --split random --seed 43"
  "--model DrugLAMPwoSSL --data human --split random --seed 44"
)

# Loop through each set of arguments and run the script
for arg in "${ARGS[@]}"
do
  echo "Running the script with argument: $arg"
  while true
  do
    python -W ignore $SCRIPT $arg --devices $devices
    exit_code=$?
    if [ $exit_code -eq 0 ]
    then
      break
    else
      echo "Error encountered. Restarting the script..."
    fi
  done
done
