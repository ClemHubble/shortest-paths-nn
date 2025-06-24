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

from refactor_training import *

output_dir = '/data/sam/terrain/'


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
    parser.add_argument('--finetune', type=int, default=0)
    parser.add_argument('--include-edge-attr', type=int, default=0)
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--finetune-from', type=str, default='none')

    args = parser.parse_args()
    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 
    aggr = args.aggr
    finetune=True if args.finetune == 1 else False
    finetune_from=None if args.finetune_from == 'none' else args.finetune_from
    trial = args.trial

    with open(args.config, 'r') as file:
        model_configs = yaml.safe_load(file)

    for modelname in model_configs:
        train_file = os.path.join(output_dir, 'data', f'{args.train_data}.npz')
        print("Training file", train_file)
        test_file = os.path.join(output_dir, 'data', args.test_data)
        

        train_data = np.load(train_file, allow_pickle=True)
        test_data = np.load(test_file, allow_pickle=True)

        train_dataset, train_node_features, train_edge_index = npz_to_dataset(train_data)

        train_edge_attr = None 
        if args.include_edge_attr:
            train_edge_attr = train_data['distances']
        print("Number of nodes:", len(train_node_features))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset, test_node_features, test_edge_index = npz_to_dataset(test_data)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)
        loss_data = []

        edge_attr = torch.tensor(train_edge_attr)
        edge_attr = edge_attr.unsqueeze(-1)
        edge_dim = 1
        graph_data = Data(x=train_node_features, edge_index=train_edge_index, edge_attr=edge_attr)        
        train_dictionary = {'graphs': [graph_data], 'dataloaders': [train_dataloader]}

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
        print(modelname, config)

        train_few_cross_terrain_case(train_dictionary=train_dictionary,
                                    model_config = config,
                                    layer_type = args.layer_type,
                                    device = args.device,
                                    epochs = args.epochs,
                                    lr= args.lr,
                                    loss_func=args.loss,
                                    aggr = aggr, 
                                    log_dir=log_dir,
                                    p = args.p,
                                    siamese=siamese,
                                    finetune_from=finetune_from)
    
if __name__=='__main__':
    main()


