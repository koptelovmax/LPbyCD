#!/bin/bash
k=5 &&
datasets=("Enzyme" "GPCR" "IC" "NR" "Kinase") &&
matchings=("com-com" "node-com") &&
for set in "${datasets[@]}"; do
   for match in "${matchings[@]}"; do
      echo Dataset "$set" matching "$match"
      python folds_generation.py "$set" > log_folds_gen_"$set".txt &&
      i=1 &&
      while [ $i -le $k ]; do
         echo Fold "$i"
         python spectral_partitioning_internal_cv.py "$set" "$match" $i > log_spectral_internal_cv_"$set"'_'"$match"'_'"$i".txt
         let i=i+1
      done
   done
done
