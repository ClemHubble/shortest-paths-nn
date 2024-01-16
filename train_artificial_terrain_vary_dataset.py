import argparse
from tqdm import tqdm, trange
from torch.optim import Adam
from torch_geometric.data import Data, HeteroData

from src.baselines import *
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.transforms import ToSparseTensor, VirtualNode, ToUndirected
from torch_geometric.nn import to_hetero
from src.transforms import add_laplace_positional_encoding, add_virtual_node
import yaml
import os
import csv

from train_baselines import *

output_dir = '/data/sam/terrain/'


def format_log_dir(output_dir, 
                   dataset_name, 
                   siamese, 
                   modelname, 
                   vn, 
                   max, 
                   loss_func, 
                   layer_type):
    log_dir = os.path.join(output_dir, 
                           'models',
                           'artificial_k_terrains', 
                           dataset_name, 
                           layer_type,
                           'vn' if vn else 'no-vn',
                           'siamese' if siamese else 'mlp')
    if not siamese:
        log_dir = os.path.join(log_dir, 'max' if max else 'sum')
    log_dir = os.path.join(log_dir, loss_func, modelname)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--config', type=str, default='configs/config-base.yml')
    parser.add_argument('--device', type=str)
    parser.add_argument('--siamese', type=int, default=0)
    parser.add_argument('--vn', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max', type=int, default=1)
    parser.add_argument('--loss', type=str, default='mse_loss')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--layer-type', type=str)
    parser.add_argument('--aggr', type=str, default='max')

    args = parser.parse_args()
    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 
    max_agg = True if args.max == 1 else False
    loss_data = []
    # Load model configs
    with open(args.config, 'r') as file:
        model_configs = yaml.safe_load(file)
    for k in range(10):
        szs = [990, 1980, 2970, 3960]
        for dataset_sz in szs:
            dataset_name = f'vary-dist-{k}-sz-{dataset_sz}'
            train_data_name = f'artificial/vary-dist/{k}-terrain-train-sz-{dataset_sz}.npz'
            test_data_name = f'artificial/vary-dist/{k}-test-100.npz'

            # Load data 
            train_file = os.path.join(output_dir, 'data', train_data_name)
            test_file = os.path.join(output_dir, 'data', test_data_name)

            train_data = np.load(train_file, allow_pickle=True)
            test_data = np.load(test_file, allow_pickle=True)

            train_dataset, train_node_features, train_edge_index = npz_to_dataset(train_data)
            print("length of train dataset", len(train_dataset))
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

            test_dataset, test_node_features, test_edge_index = npz_to_dataset(test_data)
            test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)
            print("length of test dataset", len(test_dataset))

            
            for modelname in model_configs:
                
                log_dir = format_log_dir(output_dir, dataset_name, siamese, modelname, vn, max_agg, args.loss, args.layer_type)
                config=model_configs[modelname]

                output = train_single_graph_baseline1(train_node_features, train_edge_index, train_dataloader, 
                                                    test_node_features, test_edge_index, test_dataloader,layer_type=args.layer_type, 
                                                    loss_func=args.loss, model_config = config, epochs=args.epochs, device=args.device,
                                                    siamese=siamese, log_dir=log_dir, virtual_node=vn, aggr=args.aggr, lr=args.lr)
                loss_data.append({'k': k, 'dataset_sz':dataset_sz, 'loss':output[-1]})

    # Keep track of validation losses for each configuration
    if 'base' in args.config:
        print("finished training example model")
        return 0
    modeltype = args.layer_type
    fieldnames = ['k', 'dataset_sz', 'loss']
    csv_file = os.path.join('output', 
                            'k-terrain',
                            'dataset_sz',
                            args.layer_type, 
                            'siamese' if siamese else 'mlp', 
                            'vn' if vn else 'no-vn', 
                            args.aggr, 
                            args.loss)
    if not os.path.exists(csv_file):
        os.makedirs(csv_file)
    csv_file = csv_file + f'/{modeltype}.csv'
    csvfile = open(csv_file, 'w', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in loss_data:
        writer.writerow(row)
    csvfile.close()
    print(f'Data has been written to {csv_file}')
    
if __name__=='__main__':
    main()