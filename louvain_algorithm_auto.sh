#!/bin/bash
datasets=("Enzyme" "GPCR" "IC" "NR" "Kinase") &&
for set in "${datasets[@]}"; do
   echo Louvain algorithm "$set" $1 $2
   python louvain_algorithm.py "$set" "$1" "$2" > log_louvain_"$set"'_'"$1"'_'"$2".txt
done
