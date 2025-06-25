# Repository for "De-coupled NeuroGF for Shortest Path Distance Approximation on Large Scale Terrain Graphs"
This is the corresponding repository for ""De-coupled NeuroGF for Shortest Path Distance Approximation on Large Scale Terrain Graphs" (S. Chen, P. K. Agarwal, Y. Wang, ICML 2025). 
### Dependencies
This repository depends on PyTorch and PyTorch Geometric. All code should be compatible with the latest versions of each package.

### Running 
In order to run the experiments, we use the following bash script.
`./run-experiment-1-terrain.sh <experiment-config-here> <trial>`
All experimental configurations can be found in the `experiment-configs` folder and the model configurations used can be found in the `model-configs` folder. 

## Updated MLP 06/20/2025
In our original experiments and paper, we reported results using an MLP with `LeakyReLU` as the activation function, no layer normalization, and per-layer dropout of 0.30. However, we found that we are actually able to boost the performance of just the MLP layer by using the `SiLU` activation function, layer normalization, and no dropout. 
We replicate all experiments from the paper and report results with the new MLP layer here. 
By including layer normalization and `SiLU` activation with the GAT siamese embedding module, we can also boost the approximation quality from the GAT. These experiments with the new MLP layer can be run with the same bash script but adding the `--new` flag: 

`./run-experiment-1-terrain.sh <experiment-config-here> <trial> --new`

This flag will basically turn on layer normalization, switch the activation to `SiLU` as opposed to `LeakyReLU`, and turn off dropout for the MLP layer only. To switch on layer normalization and change the activation for the GNN embedding modules, you can change them directly in the model configuration yaml. 

We include our updated results below:
### Artificial terrain results


### Norway-250 results


### All results for real terrains

#### Weighted terrains

### Terrain uncertainty modeling
