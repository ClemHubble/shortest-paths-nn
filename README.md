# Repository for "De-coupled NeuroGF for Shortest Path Distance Approximation on Large Scale Terrain Graphs"
This is the corresponding repository for ""De-coupled NeuroGF for Shortest Path Distance Approximation on Large Scale Terrain Graphs" (S. Chen, P. K. Agarwal, Y. Wang, ICML 2025). 

### Dependencies
This repository depends on PyTorch and PyTorch Geometric. All code should be compatible with the latest versions of each package.

## Running 
Experiment configurations can be found in the `experiment-configs` folder and the model configurations used can be found in the `model-configs` folder. 

### Single terrain model
In order to run the experiments for the *single terrain* case, we use the following bash script.

`./run-experiment-1-terrain.sh <experiment-config-here> <trial>`


### Cross terrain model
We recently extended our code to be able to handle training over multiple terrains. For our cross terrain case, we train each terrain sequentially per epoch. Training over multiple terrain graphs can be done using the following bash script: 

`./run-experiment-cross-terrain.sh <experiment-config-here> <trial>`

### De-coupled model
To run the de-coupled experiments, you can use another bash script. I used two separate ones for de-coupled/non de-coupled training in order to do quick changes/experiments but be warned that there's a lot duplication. There are two additional flags I use for this experiment (recently implement): `--single-terrain-per-model` and `--artificial`. If you want to train a de-coupled model on a *single* terrain, please use the `--single-terrain-per-model` flag. Otherwise, leave it out if you want to train on several different terrains at once (the cross-terrain case).

`./run-experiment-de-coupled.sh <experiment-config-here> <trial> --flag1 --flag2`.


## Updated MLP 06/20/2025
In our original experiments and paper, we reported results using an MLP with `LeakyReLU` as the activation function, no layer normalization, and per-layer dropout of 0.30. However, we found that we are actually able to boost the performance of just the MLP layer by using the `SiLU` activation function, layer normalization, and no dropout. 
We replicate all experiments from the paper and report results with the new MLP layer here. 
By including layer normalization and `SiLU` activation with the GAT siamese embedding module, we can also boost the approximation quality from the GAT. These experiments with the new MLP layer can be run with the same bash script but adding the `--new` flag: 

`./run-experiment-1-terrain.sh <experiment-config-here> <trial> --new`

This flag will basically turn on layer normalization, switch the activation to `SiLU` as opposed to `LeakyReLU`, and turn off dropout for the MLP layer only. To switch on layer normalization and change the activation for the GNN embedding modules, you can change them directly in the model configuration yaml. 

We include our updated results below (note that new GAT results use layers which incorporate layer normalization and SiLU activation):
### Artificial terrain results


### Norway-250 results
| **Method**      | **Relative error (%)** | **Absolute error (m)** |
|:----------------|:-----------------:|----------------:|
| MLP+$L_1$       | 0.71 $\pm$ 0.82   |  64.1 $\pm$ 53.8  |
| GAT+$L_1$       | 0.59 $\pm$ 0.81   | 48.2 $\pm$ 41.8   |

### All results for real terrains

Table 1. Comparing relative and absolute error with the new MLP and GAT implementations (both with layer normalization and SiLU activation) for Norway (4 million nodes) where the terrain graph is weighted by Euclidean distance between initial node embeddings. Note that we also report accuracy as the percentage of test SPD predictions with relative error below 2% and 1%. 
| **Method**      | **Relative error (%)** | **Absolute error (m)** | **Accuracy (t =0.02, %)** |**Accuracy (t =0.01, %)** |
|:----------------|:-----------------:|----------------:|:----------:|:----------:|
| Full MLP+$L_1$  | 0.73 $\pm$ 0.71   | 71.4 $\pm$ 56.7 |   91.9    |   76.3    |
| Coarse MLP+$L_1$| 0.72 $\pm$ 0.83   | 69.4 $\pm$ 66.1 |   93.3    |   76.8   |
| Coarse GAT+$L_1$| 0.62 $\pm$ 0.86   | 65.8 $\pm$ 55.5 |   96.7    |   84.8     |
| MCTR GAT        | 0.59 $\pm$ 0.81   | 41.2 $\pm$ 35.0 |   96.7    |   89.1   |


#### Weighted terrains
Table 2. Comparing relative and absolute error with the new MLP and GAT implementations (both with layer normalization and SiLU activation) for Norway (4 million nodes) where the terrain graph is weighted by 1+angle of elevation. Note that we also report accuracy as the percentage of test SPD predictions with relative error below 2% and 1%. 
| **Method**      | **Relative error (%)** | **Absolute error (m)** | **Accuracy (t =0.02, %)** |**Accuracy (t =0.01, %)** |
|:----------------|:-----------------:|----------------:|:----------:|:----------:|
| Full MLP+$L_1$  | 2.39 $\pm$ 2.09   | 204 $\pm$ 143 |   51.4    |   25.4   |
| Coarse MLP+$L_1$| 2.48 $\pm$ 2.20   | 210 $\pm$ 140 |   48.4    |   23.9   |
| Coarse GAT+$L_1$| 2.26 $\pm$ 2.30   | 184 $\pm$ 126 |   59.3    |   29.7   |
| MCTR GAT        | 1.89 $\pm$ 4.95   | 140 $\pm$ 98  |   71.0   |   41.1   |

Table 3. Comparing relative and absolute error with the new MLP and GAT implementations for LA (16 million nodes) where the terrain graph is weighted by 1+angle of elevation. 
| **Method**      | **Relative error (%)** | **Absolute error (m)** | **Accuracy (t =0.02, %)** |**Accuracy (t =0.01, %)** |
|:----------------|:-----------------:|----------------:|:----------:|:----------:|
| Full MLP+$L_1$  |    |  |      |      |
| Coarse MLP+$L_1$|    |  |      |      |
| Coarse GAT+$L_1$|    |  |      |      |
| MCTR GAT        |    |  |      |      |

### Terrain uncertainty modeling
