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

def prepare_single_terrain_dataset(train_data, batch_size):

    train_dataset, train_node_features, train_edge_index = npz_to_dataset(train_data)

    train_edge_attr = train_data['distances']
    edge_attr = torch.tensor(train_edge_attr)
    edge_attr = edge_attr.unsqueeze(-1)
    graph_data = Data(x=train_node_features, edge_index=train_edge_index, edge_attr=edge_attr) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(graph_data)
    return graph_data, train_dataloader

def get_artificial_datasets(res=2):
    dataset_names = []
    train_data_pths = []
    amps = [1.0, 2.0, 4.0, 6.0, 8.0,9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    for a in amps:
        pth = f'/data/sam/terrain/data/artificial/change-heights/amp-{a}-res-{res}-train-50k.npz'
        name =  f'/data/sam/terrain/data/artificial/change-heights/amp-{a}-res-{res}-train-50k'
        dataset_names.append(name)
        train_data_pths.append(pth)

    return dataset_names, train_data_pths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, help='Needed to construct output directory')
    parser.add_argument('--train-data', type=str)
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
    parser.add_argument('--finetune-from', type=str, nargs='+')
    parser.add_argument('--include-edge-attr', type=int, default=0)
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--single-terrain-per-model', action='store_true')
    parser.add_argument('--artificial', action='store_true')


    args = parser.parse_args()

    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 
    aggr = args.aggr
    trial = args.trial

    with open(args.config, 'r') as file:
        model_configs = yaml.safe_load(file)

    # If we are working with artificial datasets, get artificial dataset names. 
    # Train all artificial datasets sequentially together. This tends to be fast so it is 
    # easy to throw into a for-loop for training.  
    if args.artificial:
        dataset_names, train_data_pths = get_artificial_datasets(res=1)
        num_datasets = len(dataset_names)
    else:
        dataset_names = [args.dataset_name]
        train_data_pths = [os.path.join(output_dir, 'data', f'{args.train_data}.npz')]
        num_datasets = 1

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        train_data_pth = train_data_pths[i]
        prev_model_file_pth = args.finetune_from[i]
        for modelname in model_configs:

            # Provide train set dataloaders 
            if args.single_terrain_per_model:
                train_data = np.load(train_data_pth)
                graph_data, train_dataloader = prepare_single_terrain_dataset(train_data, args.batch_size)
                train_dictionary = {'graphs': [graph_data], 'dataloaders': [train_dataloader]}
            else:
                train_data = torch.load(train_data_pth)
                train_dictionary = {'graphs': train_data['graphs'], 'dataloaders': []}
                for dataset in train_data['datasets']:
                    train_dictionary['dataloaders'].append(DataLoader(dataset, batch_size = args.batch_size, shuffle=True))
            
            
            config=model_configs[modelname]
            print(modelname, config)
            model_to_finetune_from_pth = prev_model_file_pth.replace('<modelname>', modelname)
            
            train_terrains_decoupled(train_dictionary = train_dictionary,
                                    model_config = config, 
                                    layer_type = args.layer_type, 
                                    device = args.device,
                                    prev_model_pth = model_to_finetune_from_pth,
                                    finetune_dataset_name = dataset_name,
                                    epochs=args.epochs, 
                                    loss_func=args.loss,
                                    lr =args.lr,
                                    p=args.p, 
                                    aggr=aggr)
    
if __name__=='__main__':
    main()


