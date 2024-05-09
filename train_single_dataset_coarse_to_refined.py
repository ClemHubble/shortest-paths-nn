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
                   aggr, 
                   loss_func, 
                   layer_type,
                   p,
                   trial):
    log_dir = os.path.join(output_dir, 
                           'models',
                           'single_dataset', 
                           dataset_name, 
                           layer_type,
                           'vn' if vn else 'no-vn',
                           'siamese' if siamese else 'mlp',
                           f'p-{p}')
    if not siamese:
        log_dir = os.path.join(log_dir, aggr)
    log_dir = os.path.join(log_dir, loss_func, modelname, trial)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, help='Needed to construct output directory')
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--test-data', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--config', type=str, default='configs/config-base.yml')
    parser.add_argument('--device', type=str)
    parser.add_argument('--siamese', type=int, default=0)
    parser.add_argument('--vn', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--aggr', type=str, default='max')
    parser.add_argument('--loss', type=str, default='mse_loss')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--layer-type', type=str)
    parser.add_argument('--trial', type=str)
    parser.add_argument('--p', type=int, default=1 )
    parser.add_argument('--include-edge-attr', type=int, default=0)

    args = parser.parse_args()
    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 
    aggr = args.aggr
    finetune=False
    
    coarse_to_rough_datafiles = ['25x25-625.npz', 
                                 '50x50-2500.npz', 
                                 '100x100-10000.npz', 
                                 '200x200-20000.npz', 
                                 '400x400-20000.npz']
    coarse_to_rough_testfiles = ['25x25-100.npz', 
                                 '50x50-100.npz', 
                                 '100x100-100.npz', 
                                 '200x200-200.npz', 
                                 '400x400-200.npz']
    
    # coarse_to_rough_datafiles = ['25x25-525-pairs.npz', 
    #                              '50x50-2500-pairs.npz', 
    #                              '100x100-10000-pairs.npz', 
    #                              '200x200-20000.npz', 
    #                              '400x400-20000-pairs.npz']
    # coarse_to_rough_testfiles = ['25x25-100.npz', 
    #                              '50x50-250.npz', 
    #                              '100x100-250.npz', 
    #                              '200x200-400.npz', 
    #                              '400x400-100.npz']
    log_dir = format_log_dir(output_dir, 
                            args.dataset_name, 
                            siamese, 
                            'best-GNN', 
                            vn, 
                            aggr, 
                            args.loss, 
                            args.layer_type,
                            args.p,
                            args.trial)
    original_log_dir = log_dir
    for i in range(len(coarse_to_rough_datafiles)):
        # switch to off
        train_filename = coarse_to_rough_datafiles[i]
        test_filename = coarse_to_rough_testfiles[i]
        print(output_dir, train_filename)
        train_file = os.path.join(output_dir, 'data', args.train_data, train_filename)
        test_file = os.path.join(output_dir, 'data', args.test_data, 'test', test_filename)

        train_data = np.load(train_file, allow_pickle=True)
        test_data = np.load(test_file, allow_pickle=True)

        train_dataset, train_node_features, train_edge_index = npz_to_dataset(train_data)
        train_edge_attr = None 
        if args.include_edge_attr:
            train_edge_attr = train_data['distances']
        print("Number of nodes:", len(train_node_features))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

        test_dataset, test_node_features, test_edge_index = npz_to_dataset(test_data)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)

        # Load model configs
        with open(args.config, 'r') as file:
            model_configs = yaml.safe_load(file)
        
        finetune_file = os.path.join(original_log_dir, train_filename[:-4])
        if not os.path.exists(finetune_file):
            os.makedirs(finetune_file)
        if i == 0:
            # Note that log_dir is the file from which we load the previous model and the finetune file is the 
            # file where we store the final finetuned model. 
            log_dir = finetune_file
        # log_dir = '/data/sam/terrain/models/single_dataset/norway/coarse-to-rough/GeneralConvMaxAttention/vn/mlp/p-1/max/mse_loss/best-GNN/1/200x200-20000/'
        # finetune_file = '/data/sam/terrain/models/single_dataset/norway/coarse-to-rough/GeneralConvMaxAttention/vn/mlp/p-1/mse_loss/best-GNN/1/400x400-20000-pairs/'
        for modelname in model_configs:
            config=model_configs[modelname]

            train_single_graph_baseline1(train_node_features, train_edge_index, train_dataloader, 
                                        test_node_features, test_edge_index, test_dataloader,layer_type=args.layer_type, 
                                        loss_func=args.loss, model_config = config, epochs=args.epochs, device=args.device,
                                        siamese=siamese, log_dir=log_dir, virtual_node=vn, aggr=aggr, lr=args.lr, p=args.p, 
                                        log=True, finetune=finetune, edge_attr=train_edge_attr, finetune_file=finetune_file)
        log_dir = finetune_file
        finetune=True

    
if __name__=='__main__':
    main()