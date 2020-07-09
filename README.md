# Link Prediction via Community Detection in Bipartite Multi-Layer Graphs
The code for the paper presented at SAC'20:
https://dl.acm.org/doi/abs/10.1145/3341105.3373874

## The folder includes:
- **spectral_partitioning.py** - link prediction via spectral partitioning framework (input parameters: *data set name*, *matching type*, *measure type*)
- **louvain_algorithm.py** - link prediction via Louvain algorithm framework (input parameters: *data set name*, *matching type*, *measure type*)
- **spectral_partitioning_internal_cv.py** - parameter selection for spectral partitioning via internal cross-validation experiment (input parameters: *data set name*, *matching type*, *fold id*)
- **louvain_algorithm_internal_cv.py** - parameter selection for louvain algorithm via internal cross-validation experiment (input parameters: *data set name*, *matching type*, *fold id*)
- **folds_generation.py** - folds generation for parameter selection via internal cross-validation experiment (input parameters: *data set name*)

## To reconstruct experiments:
1) Unzip the arxiv with the data into 'data' folder.
2) Use **spectral_partitioning_auto.sh** for link prediction measures evaluation and parameter optimization with spectral partitioning (input parameters: *matching type*, *measure type*).
3) Use **louvain_algorithm_auto.sh** for link prediction measures evaluation and parameter optimization with Louvain algorithm (input parameters: *matching type*, *measure type*).
4) Use **spectral_partitioning_internal_cv_auto.sh** for parameter selection for spectral partitioning via internal cross-validation experiment.
5) Use **louvain_algorithm_internal_cv_auto.sh** for parameter selection for louvain algorithm via internal cross-validation experiment.
6) Retrive results from 'results' folder.

## Input parameters:
*data set name*: 'Enzyme', 'GPCR', 'IC', 'NR' or 'Kinase'

*matching type*: 'com-com' for CC matching or 'node-com' for NC matching

*measure type*: 'common neighbors', 'jaccard', 'preferential attachment', 'simrank', 'community relevance' accompanying CC matching or 'common neighbors', 'jaccard', 'preferential attachment', 'simrank', 'car neighbors', 'car jaccard', 'neighboring community' for NC matching

*fold id*: integer from 1 to 5


Environment requirements: Python 2.7, networkx 1.11, python-louvain, scikit-learn, numpy

The data can be downloaded from here: http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/, http://staff.cs.utu.fi/~aatapa/data/DrugTarget/.

The data needs to be preprocessed to adjacency lists, and ligand and drug names need to be mapped to consecutive integers, i.e. l0, l1, l2, ... , t0, t1, t2, ... .
