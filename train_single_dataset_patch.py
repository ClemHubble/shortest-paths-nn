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

output_dir = '/data/sam/terrain/'


def format_log_dir(output_dir, 
                   dataset_name, 
                   siamese, 
                   modelname, 
                   vn, 
                   aggr, 
                   loss_func, 
                   layer_type,
                   trial):
    log_dir = os.path.join(output_dir, 
                           'models',
                           'single_dataset', 
                           dataset_name, 
                           layer_type,
                           'vn' if vn else 'no-vn',
                           'siamese' if siamese else 'mlp')
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
    parser.add_argument('--include-edge-attr', type=int, default=0)

    args = parser.parse_args()
    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 
    aggr = args.aggr

    # Load data 
    print('loading data......')
    train_file = os.path.join(output_dir, 'data', args.train_data)
    test_file = os.path.join(output_dir, 'data', args.test_data)
    train_dataset = torch.load(train_file)
    test_dataset = torch.load(test_file)
    print(train_dataset[0].src, type(train_dataset[0].x))

    # DataLoader does not act correctly with np.int64 so convert to python ints
    for i in trange(len(train_dataset)):
        train_dataset[i].src = train_dataset[i].src.item()
        train_dataset[i].tar = train_dataset[i].tar.item()
        train_dataset[i].x = torch.tensor(np.array(train_dataset[i].x))
        train_dataset[i].edge_index = torch.tensor(np.array(train_dataset[i].edge_index))

    for i in trange(len(test_dataset)):
        test_dataset[i].src = test_dataset[i].src.item()
        test_dataset[i].tar = test_dataset[i].tar.item()
        test_dataset[i].x = torch.tensor(np.array(test_dataset[i].x))
        test_dataset[i].edge_index = torch.tensor(np.array(test_dataset[i].edge_index))
    

    # If virtual node, convert to TerrainHeteroData
    metadata=None
    # if args.vn:
    #     hetero_train = []
    #     for data in train_dataset:
    #         hetero_data = add_virtual_node_patch(data)
    #         hetero_train.append(hetero_data)
    #     metadata = hetero_train[0].metadata()
    #     train_dataset = hetero_train

    #     hetero_test = []
    #     for data in test_dataset:
    #         hetero_data = add_virtual_node_patch(data)
    #         hetero_test.append(hetero_data)
    #     test_dataset = hetero_test

    train_dataloader = DataLoader(train_dataset, follow_batch=['src', 'tar'], batch_size=args.batch_size)
    
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
                                args.trial)
        config=model_configs[modelname]
        print(type(train_dataloader))
        output = train_single_graph_baseline1(None, None, train_dataloader, 
                                            None, None, test_dataloader,layer_type=args.layer_type, 
                                            loss_func=args.loss, model_config = config, epochs=args.epochs, device=args.device,
                                            siamese=siamese, log_dir=log_dir, virtual_node=vn, aggr=aggr, lr=args.lr,
                                            patch=True, metadata=metadata)
        loss_data.append({'modelname':modelname, 'loss':output[-1].item()})

    # Keep track of validation losses for each configuration
    if 'base' in args.config:
        print("finished training example model")
        return 0
    modeltype = args.layer_type
    fieldnames = ['modelname', 'loss']
    csv_file = os.path.join('output', 
                            'single_dataset',
                            args.dataset_name, 
                            args.layer_type, 
                            'siamese' if siamese else 'mlp', 
                            'vn' if vn else 'no-vn', 
                            aggr, 
                            args.loss,
                            args.trial)
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