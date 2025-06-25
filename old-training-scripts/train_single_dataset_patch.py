import argparse
from tqdm import tqdm, trange
from torch.optim import Adam
from torch_geometric.data import Data, HeteroData


from src.baselines import *
from src.transforms import *
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError

from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.transforms import ToSparseTensor, VirtualNode, ToUndirected
from torch_geometric.nn import to_hetero
from src.transforms import add_laplace_positional_encoding, add_virtual_node
import yaml
import os
import csv

from train_baselines import train_single_graph_baseline1

from torch_geometric.loader import DataLoader

import time

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

class TerrainPatchesData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'src':
            return self.x.size(0)
        if key == 'tar':
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def format_patch_dataset_for_cnn():
    return 0

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
    parser.add_argument('--finetune', type=int, default=0)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--include-edge-attr', type=int, default=0)

    args = parser.parse_args()
    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 
    aggr = args.aggr

    # Load data 
    
    train_file = os.path.join(output_dir, 'data', args.train_data)
    test_file = os.path.join(output_dir, 'data', args.test_data)
    print('loading train data......')
    start = time.time()
    train_dataset = torch.load(train_file)
    end = time.time()
    print('time to load train data:', end - start)
    print("loading test  data.....")
    test_dataset = torch.load(test_file)
    print(train_dataset[0].src, type(train_dataset[0].x))

    # DataLoader does not act correctly with np.int64 so convert to python ints
    for i in trange(len(train_dataset)):
        train_dataset[i].src = train_dataset[i].src
        train_dataset[i].tar = train_dataset[i].tar
        train_dataset[i].x = torch.tensor(np.array(train_dataset[i].x))
        train_dataset[i].edge_index = torch.tensor(np.array(train_dataset[i].edge_index))
        train_dataset[i].edge_attr = torch.tensor(train_dataset[i].edge_attr)

    for i in trange(len(test_dataset)):
        test_dataset[i].src = test_dataset[i].src.item()
        test_dataset[i].tar = test_dataset[i].tar.item()
        test_dataset[i].x = torch.tensor(np.array(test_dataset[i].x))
        test_dataset[i].edge_index = torch.tensor(np.array(test_dataset[i].edge_index))
    

    # If virtual node, convert to TerrainHeteroData
    metadata=None

    train_dataloader = DataLoader(train_dataset, follow_batch=['src', 'tar'], batch_size=args.batch_size, shuffle=True)
    
    test_dataloader = DataLoader(test_dataset, follow_batch=['src', 'tar'], batch_size = args.batch_size, shuffle=True)
    
    # Load model configs
    with open(args.config, 'r') as file:
        model_configs = yaml.safe_load(file)
    
    loss_data = []
    
    for modelname in model_configs:
        log_dir = format_log_dir(output_dir, 
                                args.dataset_name, 
                                siamese, 
                                modelname, 
                                vn, 
                                aggr, 
                                args.loss, 
                                args.layer_type,
                                args.p,
                                args.trial)
        config=model_configs[modelname]
        print(type(train_dataloader))
        output = train_single_graph_baseline1(None, None, train_dataloader, 
                                            None, None, test_dataloader,layer_type=args.layer_type, 
                                            loss_func=args.loss, model_config = config, epochs=args.epochs, device=args.device,
                                            siamese=siamese, log_dir=log_dir, virtual_node=vn, aggr=aggr, lr=args.lr,
                                            patch=True, metadata=metadata, edge_attr = args.include_edge_attr, log=False)
    
if __name__=='__main__':
    main()