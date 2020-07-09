#!/bin/bash
datasets=("Enzyme" "GPCR" "IC" "NR" "Kinase") &&
for set in "${datasets[@]}"; do
   echo Spectral partitioning "$set" $1 $2
   python spectral_partitioning.py "$set" "$1" "$2" > log_spectral_"$set"'_'"$1"'_'"$2".txt
done
